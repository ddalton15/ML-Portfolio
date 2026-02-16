import speech_recognition as sr
import pyttsx3
import requests
import json


def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 200)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"⚠️ TTS failed, but text is above: {e}")


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        try:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            text = r.recognize_google(audio)
            print(f"👤 You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏰ Timeout")
            return None
        except sr.UnknownValueError:
            print("❌ Couldn't understand")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


def get_input():
    """Get input from user - either typed or spoken"""
    choice = input("\n[T]ype or [S]peak? (or 'quit'): ").strip().lower()

    if choice in ["quit", "exit", "q"]:
        return "QUIT"
    elif choice in ["t", "type", ""]:
        text = input("👤 You: ").strip()
        return text if text else None
    elif choice in ["s", "speak"]:
        return listen()
    else:
        print("Invalid choice, defaulting to type...")
        text = input("👤 You: ").strip()
        return text if text else None


def ask_ollama(prompt, context=[]):
    context.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": context,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )

        reply = ""
        print("\n🤖 AI: ", end="", flush=True)

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    text = chunk["message"]["content"]
                    print(text, end="", flush=True)
                    reply += text

        print()  # newline after response
        context.append({"role": "assistant", "content": reply})
        return reply, context
    except Exception as e:
        print(f"\n❌ Ollama error: {e}")
        return "Sorry, connection issue.", context


# Main loop
context = [
    {
        "role": "system",
        "content": "You are a LeetCode interview coach having a spoken conversation. Speak naturally like you're talking face-to-face. Never use markdown formatting - no asterisks, no numbered lists, no code blocks, no special symbols. Just speak in plain sentences as if this will be read aloud. Keep responses to 2-3 sentences. Give hints and ask questions. Don't show full code unless explicitly asked.",
    }
]

print("\n🚀 Voice + Text LeetCode Tutor")
print("=" * 50)
speak("Hi! I'm your LeetCode tutor. What problem are you working on?")

while True:
    user_input = get_input()

    if user_input == "QUIT" or (
        user_input
        and any(word in user_input.lower() for word in ["quit", "exit", "bye"])
    ):
        speak("Good luck with your coding!")
        break

    if user_input is None or user_input.strip() == "":
        print("⚠️ No input received, try again")
        continue

    reply, context = ask_ollama(user_input, context)
    speak(reply)
