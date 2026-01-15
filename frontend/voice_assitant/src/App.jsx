import { useState, useRef, useEffect } from "react";
import "./App.css";
import { FcSpeaker } from "react-icons/fc";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [recording, setRecording] = useState(false);

  const chatEndRef = useRef(null);
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  /*  WebSocket  */
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:5173/ws/voice");

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      setMessages((prev) => {
        const copy = [...prev];

        switch (data.type) {
          case "SYSTEM":
            copy.push({
              role: "assistant",
              content: data.message,
            });
            break;

          case "STT_RESULT": {
            const filtered = copy.filter((m) => !m.loading);

            filtered.push({
              role: "human",
              content: data.transcript,
            });

            return filtered;
          }

          case "ASK_AGAIN":
            copy.push({
              role: "assistant",
              content: data.prompt,
            });
            break;

          case "COMPLETE":
            copy.push({
              role: "assistant",
              content: data.message, 
            });
            break;

          default:
            break;
        }

        return copy;
      });
    };

    return () => wsRef.current?.close();
  }, []);

  /*  Voice Recording  */
  const startRecording = async () => {
    if (!wsRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);

    mediaRecorderRef.current = mediaRecorder;
    audioChunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      audioChunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, {
        type: "audio/webm",
      });

      const buffer = await audioBlob.arrayBuffer();

      setMessages((prev) => [
        ...prev,
        {
          role: "human",
          content: "กำลังแปลงเสียงเป็นข้อความ...",
          loading: true,
        },
      ]);

      wsRef.current.send(buffer);
      setRecording(false);

      stream.getTracks().forEach((t) => t.stop());
    };

    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /*  UI  */
  return (
    <div className="app-wrapper">
      <header className="app-header">
        <FcSpeaker size={40} style={{ marginRight: 8 }} />
        Voice Extraction
      </header>

      <div
        className={`chat-wrapper ${
          messages.length === 0 ? "center-input" : "has-msg"
        }`}
      >
        {messages.length > 0 && (
          <div className="chat-container">
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role}`}>
                {m.content}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}

        <div className="input-bar">
          <div className="input-bar-inner voice-bar">
            {!recording ? (
              <button className="mic-btn" onClick={startRecording}>
                🎤 กดเพื่อพูด
              </button>
            ) : (
              <button className="mic-btn recording" onClick={stopRecording}>
                ⏺️ กำลังอัดเสียง... (กดเพื่อหยุด)
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
