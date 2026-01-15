import os
import io
import json
import asyncio
import tempfile
import subprocess
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# เเปลง .webm เป็น wav เพื่อให้ gpt-4o-transcribe เเปลงจากเสียงเป็นข้อความได้
def webm_to_wav(webm_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".webm") as webm_file, \
         tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:

        webm_file.write(webm_bytes)
        webm_file.flush()

        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_file.name, wav_file.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout = 20
        )
        return wav_file.read()

class StructuredItem(TypedDict):
    field: str
    value: str | None
    หลักฐาน: str | None
    คะเเนนความลื่นไหลของข้อความ: float
    คะเเนนความต่อเนื่องของข้อความ: float
    คะเเนนความถูกต้องของข้อความ: float
    
class VoiceState(TypedDict, total=False):
    audio: bytes

    transcript: str
    stt_confidence: float
    structured: List[StructuredItem]
    final_answer: Dict[str, Any]
    missing_fields: List[str]
    final_confidence: float
    need_more: bool
    prompt: str
    output_text: Dict[str, Any]
    attempts: int

# Graph
class VoiceGraph:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.graph = self._build_graph()

    # speech to text node
    async def stt_node(self, state: VoiceState):
        print('stt_node start', flush=True)
        wav_bytes = webm_to_wav(state["audio"])

        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"

        resp = await asyncio.to_thread(
            self.client.audio.transcriptions.create,
            model="gpt-4o-transcribe",
            file=audio_file
        )
        
        return {
            "transcript": resp.text,
            "stt_confidence": 0.8,
            "attempts": state.get("attempts", 0) + 1 #เก็บไว้ว่าพูดกี่รอบเเล้ว
        }

    # ใช้ LLM (gpt-5.2) สกัดคําเเละให้คะเเนนการสะกัดคํา
    async def extract_node(self, state: VoiceState):
        print(state['transcript'],flush=True)
        prompt = f"""

ให้สกัดข้อมูลต่อไปนี้จากข้อความภาษาไทยหรืออังกฤษ หากไม่พบให้ใส่ค่าเป็น null
- ชื่อ
- นามสกุล
- เพศ
- เบอร์โทรศัพท์
- ทะเบียนรถ

กติกาเพิ่มเติม:
1. หากมีการสะกดชื่อเป็นตัวอักษร (เช่น A B C หรือ เอ บี ซี) ให้ยึดตัวอักษรที่สะกดเป็นหลัก
2. ให้ช่วยแก้คำผิดที่เกิดจากการถอดเสียง ทั้งภาษาไทยและภาษาอังกฤษ
3. "หลักฐาน" หมายถึง ข้อความช่วงที่ใช้ยืนยันว่าข้อมูลนั้นถูกสกัดออกมา
4. ให้ประเมินคะแนนในช่วง 0.0 – 1.0 โดยใช้หลักเกณฑ์ดังนี้
   - คะแนนความลื่นไหลของข้อความ: หลักฐานมีคำอื่นปนหรือขาดตอนหรือไม่
   - คะแนนความต่อเนื่องของข้อความ: การพูดต่อเนื่อง ไม่เว้นช่วงนาน
   - คะแนนความถูกต้องของข้อความ: ความถูกต้องของคำหรือรูปแบบข้อมูล
5. อย่าเดาเพศจากชื่อเเละหางเสียงครับหรือค่ะ รอให้ในข้อความมีการระบุเพศที่ชัดเจนเเล้วค่อยใส่เพศ
6. ทะเบียนรถจะต่างจากเบอร์โทรศัพท์ตรงที่ข้อความจะมาเป็นพยัญชนะเช่น ก, กอ ก่อนเเล้วตามด้วยตัวเลขหรือในบางกรณีอาจจะขึ้นต้นด้วยตัวเลขก่อน เช่น 1กง9874 โดยตัวเลขที่ติดกันจะมีไม่เกิน 4 ตัวเลข ถ้าเกินไม่ต้องบันทึก
7. เบอร์โทรศัพท์จะเป็นตัวเลข 10 ตัวเสมอถ้าไม่ครบหรือเกินไม่ต้องใส่ข้อมูลเบอร์โทรศัพท์

รูปแบบผลลัพธ์:
- ต้องเป็น JSON array เท่านั้น
- ห้ามมีข้อความอธิบายเพิ่มเติมนอก JSON
- ต้องมี object ครบทุกฟิลด์ตามโครงสร้างด้านล่าง

โครงสร้าง JSON ที่ต้องส่งกลับ:

[
  {{
    "field": ชื่อ",
    "value": "...",
    "หลักฐาน": "...",
    "คะเเนนความลื่นไหลของข้อความ": 0.0,
    "คะเเนนความต่อเนื่องของข้อความ": 0.0,
    "คะเเนนความถูกต้องของข้อความ": 0.0
  }},
  
  {{
    "field": "นามสกุล",
    "value": "...",
    "หลักฐาน": "...",
    "คะเเนนความลื่นไหลของข้อความ": 0.0,
    "คะเเนนความต่อเนื่องของข้อความ": 0.0,
    "คะเเนนความถูกต้องของข้อความ": 0.0
  }},
  
  {{
    "field": "เพศ",
    "value": "...",
    "หลักฐาน": "...",
    "คะเเนนความลื่นไหลของข้อความ": 0.0,
    "คะเเนนความต่อเนื่องของข้อความ": 0.0,
    "คะเเนนความถูกต้องของข้อความ": 0.0
  }},
  
  {{
    "field": "เบอร์โทรศัพท์",
    "value": "...",
    "หลักฐาน": "...",
    "คะเเนนความลื่นไหลของข้อความ": 0.0,
    "คะเเนนความต่อเนื่องของข้อความ": 0.0,
    "คะเเนนความถูกต้องของข้อความ": 0.0
  }},
  
  {{
    "field": "ทะเบียนรถ",
    "value": "...",
    "หลักฐาน": "...",
    "คะเเนนความลื่นไหลของข้อความ": 0.0,
    "คะเเนนความต่อเนื่องของข้อความ": 0.0,
    "คะเเนนความถูกต้องของข้อความ": 0.0
  }}
]

Transcript:
{state['transcript']}
"""


        res = await asyncio.to_thread(
            self.client.chat.completions.create,
            model="gpt-5.2",  
            messages=[
                {"role": "system", "content": "คุณคือระบบสำหรับสกัดข้อมูลเชิงโครงสร้างจากบทสนทนา"},
                {"role": "user", "content": prompt}
            ]
        )
        
        raw = res.choices[0].message.content.strip()
        print(raw, flush=True)

        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()

        try:
            parsed = json.loads(raw)

            if isinstance(parsed, list):
                structured = parsed
            elif isinstance(parsed, dict):
                structured = [parsed]
            else:
                structured = []

        except json.JSONDecodeError:
            structured = []
        
        return {"structured": structured}

    # นําคะเเนนที่ได้จาก LLM มาเช็คว่าผ่านไหม
    def check_node(self, state: VoiceState):
        structured: List[Dict[str, Any]] = state.get("structured", [])
        stt_conf = state.get("stt_confidence", 0.0)

        scores: List[float] = []

        final_answer: Dict[str, Any] = state.get(
            "final_answer",
            {
                "ชื่อ": None,
                "นามสกุล": None,
                "เพศ": None,
                "เบอร์โทรศัพท์": None,
                "ทะเบียนรถ": None
            }
        )

        for item in structured:
            if not isinstance(item, dict):
                continue

            field = item.get("field")
            value = item.get("value")

            if field not in final_answer:
                continue

            field_scores = [
                item.get("คะเเนนความลื่นไหลของข้อความ", 0.0),
                item.get("คะเเนนความต่อเนื่องของข้อความ", 0.0),
                item.get("คะเเนนความถูกต้องของข้อความ", 0.0),
            ]

            avg_field_score = sum(field_scores) / len(field_scores)
            final_field_score = stt_conf * 0.5 + avg_field_score * 0.5
            scores.append(final_field_score)

            # ไม่ผ่านหรือ field นั้นมีอยู่เเล้วให้ข้าม
            if value in (None, "", "null") or final_field_score < 0.7:
                continue

            # ผ่านให้ update ค่า
            final_answer[field] = value

        # ดูว่าขาด field ไหนบ้าง
        missing_fields = [
            field for field, value in final_answer.items()
            if value is None
        ]

        need_more = bool(missing_fields)

        final_confidence = (
            sum(scores) / len(scores)
            if scores else 0.0
        )

        prompt = (
            f"ยังขาดข้อมูล {', '.join(missing_fields)} กรุณาพูดเพิ่มเติม"
            if need_more else ""
        )

        return {
            "final_answer": final_answer,
            "missing_fields": missing_fields,
            "final_confidence": final_confidence,
            "need_more": need_more,
            "prompt": prompt
        }


    #ดูว่าขาดข้อมูล field ไหนบ้างไหม
    def router(self, state: VoiceState):
        return "ASK_AGAIN" if state.get("need_more") else "Done"
    
    #ถ้าข้อมูลครบเเล้วให้เเสดงผล
    def final_node(self, state: VoiceState):
        output = state.get("final_answer", {})
        
        lines = []
        for field, value in output.items():
            if field in ["ชื่อ","นามสกุล"] :
                lines.append(f"{field}: {value.capitalize() or '-'}")
            else :
                lines.append(f"{field}: {value or '-'}")
        
        return {
        "output_text": "ข้อมูลของคุณ\n\n" + "\n".join(lines)
        }

    def _build_graph(self):
        graph = StateGraph(VoiceState)

        graph.add_node("stt", self.stt_node)
        graph.add_node("extract", self.extract_node)
        graph.add_node("check", self.check_node)
        graph.add_node("final", self.final_node)

        graph.set_entry_point("stt")
        graph.add_edge("stt", "extract")
        graph.add_edge("extract", "check")

        graph.add_conditional_edges(
            "check",
            self.router,
            {
                "ASK_AGAIN": END,
                "Done": "final"
            }
        )

        graph.add_edge("final", END)

        return graph.compile()

    async def run(self, state: VoiceState):
        return await self.graph.ainvoke(state)
