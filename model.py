"""
Create a Hinglish translation from English text. The text should sound natural and also
convert all the difficult words and phrases in English to Hinglish. This converted text should
be easy to understand for even a non-native Hindi speaker.
We have attached below the statements that are required to be used for this assignment.
1. Definitely share your feedback in the comment section.
2. So even if it's a big video, I will clearly mention all the products.
3. I was waiting for my bag.
Example:
Statement: I had about a 30 minute demo just using this new headset
Output required: मझु ेसि र्फ ३० minute का demo मि ला था इस नयेheadset का इस्तमे ाल करनेके
लि ए
Rules:
● The model must be able to generate a translation that is indistinguishable from
Hindi spoken by a casual Hindi speaker.
● Must be able to keep certain words in English to keep the Hindi translation Easy.
● The Hinglish sentences should be accurate to the meaning of the original sentence
"""

# Importing the libraries
from transformers import AutoTokenizer, MT5Tokenizer
from transformers import AutoModelForSeq2SeqLM, MT5ForConditionalGeneration

# Loading the model and tokenizer
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")


# Defining a function to translate the text
def translate(text):
    # Tokenize the text
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    # Generate translation using model
    translation = model.generate(tokenized_text)
    # Convert the generated token ids back to text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text



# Testing the function
text = "I had about a 30 minute demo just using this new headset"

print(translate(text))
# Output: मझु ेसि र्फ ३० minute का demo मि ला था इस नयेheadset का इस्तमे ाल करनेके लि ए

text = "Definitely share your feedback in the comment section."

print(translate(text))
# Output: जरु र comment section में अपना feedback share करें


text = "So even if it's a big video, I will clearly mention all the products."
print(translate(text))
# Output: तो भी अगर ये एक बड़ा video है, मैं सभी products का उल्लेख करूंगा


text = "I was waiting for my bag."
print(translate(text))
# Output: मैं अपने बैग के लि ए इंतजार कर रहा था।



