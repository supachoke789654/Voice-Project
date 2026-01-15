import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent_runner import VoiceGraph

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = FastAPI()
voice_graph = VoiceGraph()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#ใช้ Websocket
@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()

    state = {
        "attempts": 0
    }

    await ws.send_json({
        "type": "SYSTEM",
        "message": "กรุณากดปุ่มเพื่อพูดข้อมูลของคุณ"
    })

    try:
        while True:
            audio = await ws.receive_bytes()
            if not audio:
                continue

            state["audio"] = audio

            result = await voice_graph.run(state)
            state.update(result)

            # เเสดงผล Speech to text ก่อน
            if "transcript" in result:
                await ws.send_json({
                    "type": "STT_RESULT",
                    "transcript": result["transcript"],
                    "stt_confidence": result.get("stt_confidence", 0.0)
                })
                await asyncio.sleep(0.3)

            # ในกรณีที่ต้องถามเพิ่ม
            if result.get("need_more", False):
                await ws.send_json({
                    "type": "ASK_AGAIN",
                    "prompt": result["prompt"],
                    "missing": result.get("missing_fields", []),
                    "confidence": result.get("final_confidence", 0.0)
                })
                
            else:
                #ถ้าข้อมูลครบให้ print
                await ws.send_json({
                    "type": "COMPLETE",
                    "message": result.get("output_text", ""),
                    "confidence": result.get("final_confidence", 0.0)
                })
                break

    except WebSocketDisconnect:
        print("WebSocket client disconnected")


