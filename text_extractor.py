import pytesseract
from PIL import Image
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import os

# Load an image of a historical document
image_path = r"C:\Suprith\kl\assignments\historical_file.jpg"
image = Image.open(image_path)

# Extract text from the image using OCR
text = pytesseract.image_to_string(image, lang='eng')

print("----- Extracted Text -----")
print(text[:500])

# Basic cleaning
text = text.replace('\n', ' ').strip()

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Analyze named entities
print("\n----- Named Entities -----")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Count the most common words (ignoring stopwords and punctuation)
words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
common_words = Counter(words).most_common(10)

# Plot top words
plt.bar([w[0] for w in common_words], [w[1] for w in common_words])
plt.title("Most Common Words in Historical Document")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()

# Save extracted text for later study
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
