import whisper

model = whisper.load_model("base")
result = model.transcribe("file.mp3")
print(result["text"])