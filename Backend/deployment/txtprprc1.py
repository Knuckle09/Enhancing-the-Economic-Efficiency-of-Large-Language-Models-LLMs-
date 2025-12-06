import nltk
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk import pos_tag

class TextPreprocessor:
    def __init__(self, language='english', download_resources=True):
        if download_resources:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        self.language = language
        self.stopwords = set(stopwords.words('english' if language == 'english' else 'portuguese'))
        
        self.high_similarity_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
        }
        
        self.lemmatizer = WordNetLemmatizer()
        self.synonym_cache = {}

        self.math_terms = {
            "polynomial", "function", "graph", "maximum", "minimum", 
            "input", "output", "point", "rate", "change", "zero", 
            "determine", "intervals", "increases", "decreases", "shape",
            "equation", "variable", "turn", "turning", "behavior", "curve",
            "infinity", "infinite", "local", "global", "axis", "coordinate"
        }
        
        self.technical_terms = {
            "software", "system", "design", "algorithm", "data", "code",
            "implement", "detection", "processing", "complex", "test", "tester",
            "programming", "development", "application", "server", "client",
            "database", "query", "interface", "framework", "library", "API",
            "function", "class", "method", "object", "instance", "inheritance",
            "polymorphism", "encapsulation", "abstraction", "template",
            "debugging", "compile", "runtime", "syntax", "semantic"
        }

        self.code_terms = {
            "variable", "function", "method", "class", "object", "array",
            "list", "dict", "dictionary", "set", "tuple", "string", "int",
            "float", "boolean", "true", "false", "null", "nil", "undefined",
            "import", "export", "require", "include", "module", "package",
            "library", "framework", "api", "endpoint", "request", "response",
            "http", "get", "post", "put", "delete", "patch", "async", "await",
            "promise", "callback", "event", "listener", "emit", "trigger",
            "parameter", "argument", "return", "yield", "throw", "catch", "try",
            "except", "finally", "error", "exception", "loop", "iterate", "for",
            "while", "do", "break", "continue", "if", "else", "switch", "case",
            "default", "regex", "pattern", "match", "search", "replace"
        }

        self.generic_important_terms = {
            "create", "generate", "build", "make", "design", "develop",
            "implement", "process", "analyze", "execute", "run", "perform",
            "explain", "describe", "detail", "outline", "summarize", "write",
            "compose", "craft", "produce", "construct", "formulate", "devise",
            "imagine", "think", "consider", "evaluate", "assess", "examine",
            "inspect", "review", "check", "verify", "validate", "test",
            "debug", "troubleshoot", "fix", "solve", "resolve", "address",
            "tackle", "handle", "manage", "organize", "arrange", "structure"
        }
        
        self.domain_terms = self.math_terms.union(
            self.technical_terms, 
            self.code_terms,
            self.generic_important_terms
        )
        
        self.important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'RB'}

    def preprocess(self, text, verbose=False, preserve_semantics=True, high_similarity=True):
        if verbose:
            print("Original text:", text)
        
        if self._is_likely_code_prompt(text):
            return self._preprocess_code_prompt(text, verbose)
            
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        stopwords_to_use = self.high_similarity_stopwords if high_similarity else self.stopwords
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            tagged_tokens = pos_tag(tokens) if high_similarity else []
            pos_dict = {t[0]: t[1] for t in tagged_tokens} if high_similarity else {}
            
            processed_tokens = []
            
            bigrams_list = list(ngrams(tokens, 2)) if high_similarity else []
            bigrams_phrases = [' '.join(bg).lower() for bg in bigrams_list]
            
            tokens_in_kept_bigrams = set()
            important_bigrams = []
            
            if high_similarity:
                for i, bigram in enumerate(bigrams_list):
                    bg_lower = ' '.join(bigram).lower()
                    if (any(term in bg_lower for term in self.domain_terms) or
                        (pos_dict.get(bigram[0], '').startswith('NN') and pos_dict.get(bigram[1], '').startswith('NN'))):
                        important_bigrams.append(bigram)
                        tokens_in_kept_bigrams.add(bigram[0])
                        tokens_in_kept_bigrams.add(bigram[1])
            
            for bigram in important_bigrams:
                processed_tokens.append(' '.join(bigram).lower())
            
            for token in tokens:
                if token in tokens_in_kept_bigrams:
                    continue
                    
                token_lower = token.lower()
                
                is_structural = not token.isalnum() and token not in [',', '.', ';', ':', '!', '?']
                
                if preserve_semantics and token_lower in self.domain_terms:
                    processed_tokens.append(token_lower)
                    continue
                
                if token.isdigit() or bool(re.match(r'^-?\d+(\.\d+)?$', token)):
                    processed_tokens.append(token)
                    continue
                
                if high_similarity and pos_dict.get(token, '') in self.important_pos:
                    lemma = self.lemmatizer.lemmatize(token_lower)
                    processed_tokens.append(lemma)
                    continue
                
                if token_lower in stopwords_to_use or not token.isalnum():
                    if high_similarity and token_lower in {'with', 'using', 'by', 'as', 'into', 'from'}:
                        processed_tokens.append(token_lower)
                    continue
                
                lemma = self.lemmatizer.lemmatize(token_lower)
                
                if preserve_semantics and not high_similarity and len(lemma) > 6:
                    shortest = self.get_shortest_synonym(lemma)
                    if shortest and shortest != lemma and len(shortest) < len(lemma):
                        processed_tokens.append(shortest)
                    else:
                        processed_tokens.append(lemma)
                else:
                    processed_tokens.append(lemma)
            
            if processed_tokens:
                processed_sentences.append(" ".join(processed_tokens))
        
        result = " ".join(processed_sentences)
        
        if verbose:
            print("Processed text:", result)
            
        return result

    def _is_likely_code_prompt(self, text):
        code_indicators = [
            "function", "class", "method", "algorithm", "implement",
            "code", "program", "script", "syntax", "write a", 
            "implement a", "create a", "def ", "function", "return",
            "```", "import", "from", "public class", "int ", "void ",
            "#include", "println", "console.log", "print(", "printf"
        ]
        
        code_term_count = sum(1 for term in self.code_terms if term in text.lower())
        
        has_code_formatting = "```" in text or "    " in text
        
        return (any(indicator in text.lower() for indicator in code_indicators) or 
                code_term_count >= 3 or 
                has_code_formatting)

    def _preprocess_code_prompt(self, text, verbose=False):
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            processed_tokens = []
            
            for token in tokens:
                token_lower = token.lower()
                
                if token_lower in {'a', 'an', 'the', 'is', 'are', 'was', 'were'}:
                    continue
                
                if token_lower in self.code_terms or token.isalnum():
                    processed_tokens.append(token_lower)
                    continue
                
                if token in {'.', '(', ')', '[', ']', '{', '}', ':', ';', ',', '+', '-', '*', '/', '=', '<', '>'}:
                    processed_tokens.append(token)
                    continue
                
                if not token.isalnum():
                    continue
                
                processed_tokens.append(token_lower)
            
            if processed_tokens:
                processed_sentences.append(" ".join(processed_tokens))
        
        result = " ".join(processed_sentences)
        
        if verbose:
            print("Processed code prompt:", result)
            
        return result

    def get_shortest_synonym(self, word, min_similarity=0.8):
        if word in self.synonym_cache:
            return self.synonym_cache[word]
            
        synsets = wn.synsets(word)
        if not synsets:
            self.synonym_cache[word] = word
            return word
            
        original_synset = synsets[0]  
        similar_synonyms = {}
        
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) < len(word):
                    for syn_synset in wn.synsets(synonym):
                        similarity = original_synset.wup_similarity(syn_synset)
                        if similarity and similarity >= min_similarity:
                            similar_synonyms[synonym] = similarity
        
        if similar_synonyms:
            result = min(similar_synonyms.keys(), key=len)
        else:
            result = word
            
        self.synonym_cache[word] = result
        return result

    def aggressive_preprocess(self, text, target_reduction=0.5):
        
        aggressive_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'of', 'to', 'in', 'for', 'with', 'by', 'from',
            'at', 'on', 'off', 'over', 'under', 'above', 'below', 'up', 'down',
            'please', 'kindly', 'could you', 'would you', 'can you'
        }
        
        import re
        redundant_patterns = [
            r'\b(please|kindly|could you|would you|can you)\b',
            r'\b(i need|i want|i would like)\b',
            r'\b(make sure|ensure that|be sure to)\b',
            r'\b(in order to|so as to)\b',
            r'\b(it is important to|it is necessary to)\b'
        ]
        
        processed_text = text.lower()
        for pattern in redundant_patterns:
            processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        tokens = word_tokenize(processed_text)
        pos_tags = pos_tag(tokens)
        
        essential_tokens = []
        for token, pos in pos_tags:
            if (token in self.math_terms or token in self.code_terms or 
                token in self.technical_terms or token.isdigit()):
                essential_tokens.append(token)
            elif pos.startswith(('NN', 'VB', 'JJ')) and token not in aggressive_stopwords:
                essential_tokens.append(token)
        
        return ' '.join(essential_tokens)

    def balanced_preprocess(self, text, target_reduction=0.45, min_similarity=0.80):
        filler_patterns = [
            r'\b(please|kindly|could you|would you|can you)\b',
            r'\b(think carefully and logically)\b',
            r'\b(step by step)\b',
            r'\b(as follows|the following)\b',
            r'\b(in detail|carefully)\b'
        ]
        
        processed = text
        for pattern in filler_patterns:
            processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
        
        compressions = {
            'function signature is': 'def',
            'the function signature is': 'def',
            'return': '→',
            'given': '→',
            'calculate': 'calc',
            'determine': 'find',
            'implement': 'code',
            'create': 'make'
        }
        
        for verbose, short in compressions.items():
            processed = re.sub(rf'\b{re.escape(verbose)}\b', short, processed, flags=re.IGNORECASE)
        
        words = word_tokenize(processed.lower())
        pos_tags = pos_tag(words)
        
        important_words = []
        for word, pos in pos_tags:
            if (pos.startswith(('NN', 'VB', 'JJ', 'CD')) or 
                word.isdigit() or 
                word in self.domain_terms or
                len(word) <= 2):
                important_words.append(word)
        
        return ' '.join(important_words)

    def _optimize_math_prompt(self, prompt):
        prompt = prompt.lower()
        prompt = re.sub(r'\b(the sum of|add|plus)\b', '+', prompt)
        prompt = re.sub(r'\b(minus|subtract|less)\b', '-', prompt)
        prompt = re.sub(r'\b(times|multiplied by|product of)\b', '*', prompt)
        prompt = re.sub(r'\b(divided by|over|quotient of)\b', '/', prompt)
        prompt = re.sub(r'\b(equals|is equal to|is)\b', '=', prompt)
        tokens = word_tokenize(prompt)
        tokens = [w for w in tokens if w not in self.stopwords and w.isalnum()]
        return ' '.join(tokens)

    def _optimize_coding_prompt(self, prompt):
        prompt = re.sub(r'(explain|describe|step by step|as a developer|please|write|implement|create|show|give|find|how to|can you|could you|would you|generate|provide|draft|compose|build|construct|develop|produce|demonstrate|list|print|output|return|calculate|determine|solve|define|write a function to|write a program to)', '', prompt, flags=re.IGNORECASE)
        tokens = word_tokenize(prompt)
        tokens = [w for w in tokens if w.lower() not in self.stopwords]
        return ' '.join(tokens)

    def _optimize_business_prompt(self, prompt):
        prompt = re.sub(r'(please|kindly|could you|would you|i would like to|i am writing to|this is to inform you|dear|regards|sincerely|thank you|thanks)', '', prompt, flags=re.IGNORECASE)
        tokens = word_tokenize(prompt)
        tokens = [w for w in tokens if w.lower() not in self.stopwords]
        return ' '.join(tokens)

    def aggressive_optimize(self, prompt, category=None):
        tokens = word_tokenize(prompt)
        tagged = pos_tag(tokens)
        keep = [w for w, t in tagged if t.startswith(('NN', 'VB')) or w.isdigit()]
        keep = [w for w in keep if w.lower() not in self.stopwords]
        if category == "math":
            prompt = prompt.replace("the sum of", "+").replace("is equal to", "=")
        elif category == "coding":
            prompt = prompt.replace("explain", "").replace("step by step", "")
        return ' '.join(keep)

def main(text, language='english'):
    preprocessor = TextPreprocessor(language=language)
    processed_text = preprocessor.preprocess(text, high_similarity=True)
    
    return processed_text

if __name__ == "__main__":
    text = "Imagine you are a seasoned software tester tasked with ensuring the quality and reliability of a new, complex e-commerce website"
    main(text)
    pass
