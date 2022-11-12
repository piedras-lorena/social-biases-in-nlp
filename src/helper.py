from google.cloud import language_v1

def predict_sentiment(sentence):
    client = language_v1.LanguageServiceClient()

    # The text to analyze
    document = language_v1.Document(
        content=sentence, type_=language_v1.Document.Type.PLAIN_TEXT
    )

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    print("Text: {}".format(sentence))
    print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))
    return sentiment