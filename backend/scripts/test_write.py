
print("Start")
with open("backend/scripts/test_output.txt", "w", encoding="utf-8") as f:
    f.write("Line 1\n")
    print("Importing backend...")
    try:
        from backend.features.sentiment import sentiment_analyzer
        f.write("Imported backend.\n")
    except Exception as e:
        f.write(f"Import failed: {e}\n")
    
    f.write("Line 2\n")

print("Done")
