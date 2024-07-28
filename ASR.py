import whisper

model = whisper.load_model("base")
result = model.transcribe("Data/file.mp3")
print(result["text"])