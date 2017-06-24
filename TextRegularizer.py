import re
#from stemming.porter2 import stem # FIXME: either use this or delete it

sad_emoticon_pattern=re.compile(
        r"[\:\;\=][\w'-\*-]*[\(\[\{\\\/<\|@]|[\)\]\}\\\/<\|][\w'-\*-]*[\:\;\=]",
        re.M | re.DOTALL
    )

happy_emoticon_pattern=re.compile(
        r"[\:\;\=][\w'-\*-]*[\)\]\}]|[\(\[\{][\w'-\*-]*[\:\;\=]",
        re.M | re.DOTALL
    )

haha_pattern=re.compile(
        r"a+h+a+.{0,3}|h+a+h.{0,3}|e+h+e+.{0,3}|h+e+h.{0,3}",
        re.M | re.DOTALL
    )

happybirthday_pattern=re.compile(
        r"happybirthday.*",
        re.M | re.DOTALL
    )

time_pattern=re.compile(
        r"[0-9:\.]+[amp]+",
        re.M | re.DOTALL
    )

date_pattern=re.compile(
        r"(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2])\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)0?2\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9])|(?:1[0-2]))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})",
        re.M | re.DOTALL
    )

def _convert_sad_emoticons(word):
    if sad_emoticon_pattern.match(word):
        return True, [':(']
    else:
        return False, None

def _convert_happy_emoticons(word):
    if happy_emoticon_pattern.match(word):
        return True, [':)']
    else:
        return False, None

def _convert_haha(word):
    if haha_pattern.match(word):
        return True, ['haha']
    else:
        return False, None

def _convert_happy_birthday(word):
    if happybirthday_pattern.match(word):
        return True, ['happybirthday']
    else:
        return False, None

def _convert_hashtag(word, word_to_occurrence):
    if word.startswith('#'):
        big_word=word[1:]
        match_found=False

        def find_subwords(from_index, end_index):
            if from_index==end_index:
                return []

            subwords=[]

            while len(subwords)==0 and from_index<end_index-1:
                current_subwords=[]

                to_index=end_index
                while not from_index==to_index:
                    current_word=big_word[from_index:to_index]
                    # print("Comparing %s" %current_word)
                    if current_word in word_to_occurrence:
                        # print("Found word %s" %current_word)
                        current_subwords.append(current_word)
                        current_subwords+=find_subwords(to_index, len(big_word))
                        break
                    else:
                        to_index=to_index-1

                if len(current_subwords)<len(subwords) or len(subwords)==0:
                    subwords=current_subwords[:]
                from_index=from_index+1

            return subwords

        word_list=find_subwords(0, len(big_word))
        # print("Replacing %s with" %word)
        # print(word_list)
        return True, word_list
    else:
        return False,None

# # Alternative implementation - just for completeness of the codebase during development
# def recursive_hashtag_tokenization(self, hashtag):
#     def find_tokenization(tweet):
#         tokenization = []
#         occurrence = 0
#         for begin in range(3):
#             for end in range(begin + 2, len(tweet) + 1):
#                 if tweet[begin:end] in self.word_to_id and \
#                         (len(tweet[begin:end]) > 3 or self.word_to_occurrence[tweet[begin:end]] > 50) and \
#                         (len(tweet[begin:end]) > 2 or self.word_to_occurrence[tweet[begin:end]] > 200):
#                     tokens, accum_occurrence = find_tokenization(tweet[end:])
#                     if len(tokens) == 0 and len(tweet[end:]) > 3:
#                         continue
#                     if (end - begin) + accum_occurrence > occurrence:
#                         tokenization = [tweet[begin:end]] + tokens
#                         occurrence = (end - begin) + accum_occurrence
#         return tokenization, occurrence
#
#     tokenized_hashtag, accum_occurrence = find_tokenization(hashtag.lstrip('#'))
#     print("Tokenizing hashtag... \t{}  -->   \t[{}]\n".format(hashtag, ', '.join(tokenized_hashtag)))
#     return tokenized_hashtag, accum_occurrence


def _tag_number(word):
    number_of_digits = sum(c.isdigit() for c in word)
    if number_of_digits>3 and len(word)<=number_of_digits+3:
        return True, ['number']   # replace number by tag
    else:
        return False, []

class TextRegularizer:
    def __init__(self, vocabulary):
        self.regularizing_functions=[_convert_sad_emoticons,
                                     _convert_happy_emoticons,
                                     _convert_haha,
                                     _convert_happy_birthday,
                                     _tag_number,
                                     lambda word: _convert_hashtag(word, vocabulary.word_to_occurrence)]


    def regularize_word_vocab(self, word, word_to_occurrence=None):
        for regularizing_func in self.regularizing_functions[:-1]:
            matched, new_word=regularizing_func(word)
            if matched:
                return new_word
        if word_to_occurrence is not None:
            matched, new_word = _convert_hashtag(word, word_to_occurrence)
            if matched:
                return new_word
        return [word]

    def regularize_word(self, word):
        for regularizing_func in self.regularizing_functions:
            matched, new_word=regularizing_func(word)
            if matched:
                return new_word
        return [word]

    def get_special_words(self):
        '''
        Used to extend dictionary with replacement symbols that will be generated by regularizer
        :return: list of replacement symbols
        '''
        return [':(',':)','haha','happybirthday', 'time', 'date', 'number']