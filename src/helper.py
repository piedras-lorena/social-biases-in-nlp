from google.cloud import language_v1
import pandas as pd

def predict_sentiment(sentence, debug=False):
    client = language_v1.LanguageServiceClient()

    # The text to analyze
    document = language_v1.Document(
        content=sentence, type_=language_v1.Document.Type.PLAIN_TEXT
    )

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(
        request={"document": document}
    )#.document_sentiment

    if debug:
        print("Text: {}".format(sentence))
        print("Sentiment: {}, {}".format(sentiment.score, sentiment.magnitude))
        
    return sentiment



def sample_analyze_sentiment(text_content, debug=False):
    """
    Analyzing Sentiment in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'I am so happy and joyful.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    
    if debug:
        # Get overall sentiment of the input document
        print(u"Document sentiment score: {}".format(response.document_sentiment.score))
        print(
            u"Document sentiment magnitude: {}".format(
                response.document_sentiment.magnitude
            )
        )
        # Get sentiment for all sentences in the document
        for sentence in response.sentences:
            print(u"Sentence text: {}".format(sentence.text.content))
            print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
            print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))

        # Get the language of the text, which will be the same as
        # the language specified in the request or, if not specified,
        # the automatically-detected language.
        print(u"Language of the text: {}".format(response.language))
        
    return response.sentences[0].sentiment.score

def convert_perturbed_to_long(data):    
    data_long = data.melt(
        id_vars=['op_gender','subreddit', 'original', 'category'], 
        value_vars=['recommended_sentence', 'non_recommended_sentence'],
        value_name='sentence'
        )

    original_data = data.filter(['subreddit', 'original', 'op_gender']).drop_duplicates()
    original_data = original_data.assign(
        sentence=original_data.original,
        variable='original_sentence',
        category='ORIGINAL'
    )
    data_long = pd.concat([data_long, original_data])
    data_long = data_long.sort_values('original').reset_index(drop=True)

    data_long = data_long.reset_index().rename(
        columns={'index':'id'})
    return data_long