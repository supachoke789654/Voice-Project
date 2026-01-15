# ระบบสกัดข้อมูลจากเสียง (Voice Extraction)

โปรเจกต์นี้เป็นระบบสกัดข้อมูลจากเสียง  
โดยแบ่งสถาปัตยกรรมออกเป็น Frontend และ Backend ดังนี้

Frontend พัฒนาด้วย **React (Vite)**  
Backend พัฒนาด้วย **FastAPI** พร้อมรองรับ **WebSocket** และใช้ **LangGraph** ในการควบคุมลำดับการทำงาน

---

## LangGraph Architecture

ระบบประกอบด้วย LangGraph Nodes ดังนี้

1. **Speech to Text Node**  
   ใช้ gpt-4o-transcibe ในการสกัดข้อความจากเสียง ซึ่ง
   รองรับทั้งภาษาไทย อังกฤษเเละภาษาไทยปนอังกฤษ

2. **LLM Extraction Node**  
   ใช้ LLM สะกัดข้อมูลจากข้อความที่ได้จาก Speech to Text Node
   โดยใช้โมเดล gpt-5.2 เเละจะมีการให้ LLM ให้คะเเนนข้อมูลที่สกัดมาด้วยว่ามีความมั่นใจมากเเค่ไหน

3. **Checking Node**  
   Node นี้มีหน้าที่ในการเช็คข้อมูลที่สกัดได้จาก LLM Extraction Node ว่ามีข้อมูลส่วนไหนที่ไม่สามารถสกัดได้
   เเละมีข้อมูลส่วนไหนที่ไม่ชัดเจนโดยจะมีการเเจ้งกลับไปว่าขาดข้อมูลไหนบ้าง

4. **Output Node**  
   เป็น Node ที่จะทําการเเสดงให้ User เห็นข้อมูลทั้งหมดของตน เมื่อ User เเจ้งข้อมูลครบเเล้ว

---

## Running with Docker

เมื่อรันระบบด้วย Docker Compose สามารถเข้าถึงบริการต่าง ๆ ได้ดังนี้

- **Frontend**  
  http://localhost:5173

- **Backend API**  
  http://localhost:8000

---

## Test Audio

1. **Complete Information**  
   เป็น Test Case ที่ User เเจ้งข้อมูลตัวเองครบถ้วน

2. **Incomplete Information**  
   เป็น Test Case ที่ User เเจ้งข้อมูลตัวเองไม่ครบ

3. **Thai & English**  
   เป็น Test Case ที่ User เเจ้งข้อมูลของตนเองเป็นภาษาไทยปนอังกฤษ
