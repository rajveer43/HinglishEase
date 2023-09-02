import googletrans
from googletrans import Translator


def translate_to_hinglish(text):
    translator = Translator(service_urls=["translate.google.com"])
    translation = translator.translate(text, src="en", dest="hi")
    return translation.text


def main():
    sentences = [
        "I had about a 30-minute demo just using this new headset.",
        "So even if it's a big video, I will clearly mention all the products.",
        "I was waiting for my bag.",
    ]

    print("Original Sentences:")
    for sentence in sentences:
        print(sentence)

    print("\nTranslations to Hinglish:")
    for sentence in sentences:
        hinglish_text = translate_to_hinglish(sentence)
        print(hinglish_text)


if __name__ == "__main__":
    main()