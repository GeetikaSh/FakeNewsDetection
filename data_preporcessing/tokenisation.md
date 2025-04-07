# Tokenisation
Tokenization is the process of breaking down text into smaller pieces called tokens. Tokens can be words, characters, or subwords, depending on how granular you want the processing to be.

## Example
Tweet: "Fake news spreads faster than real news!"
Result: ['Fake', 'news', 'spreads', 'faster', 'than', 'real', 'news', '!']

##  Why Do We Need Tokenization in Your Project?
We are working on **Fake News Detection**, so we are likely to build a model but models don't understand raw text as they only understant the numbers. Tokenisation is the fist step in coverting text into something the model can understand.

**Key Reasons:**
- Converts text to a form usable by ML models
- Helps you analyze patterns in text (e.g., frequency of certain words)
- Used for vectorization (like Bag of Words, TF-IDF, or embeddings)
- Supports cleaning, normalization, and preprocessing