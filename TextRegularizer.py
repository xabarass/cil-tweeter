import re

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
        return True, [TextRegularizer.tags['sad_emoticon']]
    else:
        return False, None

def _convert_happy_emoticons(word):
    if happy_emoticon_pattern.match(word):
        return True, [TextRegularizer.tags['happy_emoticon']]
    else:
        return False, None

def _convert_haha(word):
    if haha_pattern.match(word):
        return True, [TextRegularizer.tags['haha']]
    else:
        return False, None

def _convert_happy_birthday(word):
    if happybirthday_pattern.match(word):
        return True, [TextRegularizer.tags['happybirthday']]
    else:
        return False, None

def _convert_hashtag(word, word_to_occurrence):
    if word.startswith('#') and not word in word_to_occurrence:
        big_word=word[1:]
        match_found=False

        find_subwords_cache = {}


        # Find optimal tokenization of big_word[from_index:end_index] of maximum size max_len
        # Return success status, tokenization, new_max_len, score
        def find_subwords(from_index, end_index, max_len, tolerance):

            # Tokenization exceeds maximum length
            if max_len < 0 or (max_len == 0 and from_index + tolerance < end_index ):
                return False, [], 0, 0

            # Base case
            if from_index == end_index or (max_len == 0 and from_index + tolerance >= end_index):
                return True, [], 0, 0

            base_from_index = from_index

            if base_from_index in find_subwords_cache:
                cached_result = find_subwords_cache[base_from_index]
                if cached_result[2] <= max_len:
                    return cached_result[0], cached_result[1], cached_result[2], cached_result[3]
                else:
                    return False, [], 0, 0

            success_flag = False
            best_tokenization=[]
            best_score = 0

            while from_index < end_index:

                to_index = end_index

                while from_index != to_index:
                    if max_len == 1 and to_index + tolerance < end_index: # tokenization cannot be completed (missing big_word[to_index:end_index-1]) as tolerance is 1
                        break
                    current_token = big_word[from_index:to_index]
                    if current_token in word_to_occurrence:
                        # print("Found word %s" %current_token)
                        current_token_score = word_to_occurrence[current_token]*len(current_token)
                        # current_token_len = len(current_token)
                        # if current_token_score*current_token_len < best_token_score*best_token_len:
                        #     continue
                        child_success_flag, child_tokenization, child_len, child_score = find_subwords(to_index, end_index, max_len-1, tolerance)
                        assert child_len < max_len
                        if child_success_flag and \
                                ((not success_flag) or
                                  child_len + 1 < max_len or
                                 (child_len + 1 == max_len and child_score + current_token_score > best_score)):

                            best_tokenization = [current_token]
                            best_tokenization += child_tokenization
                            max_len = len(best_tokenization)
                            best_score = child_score + current_token_score
                            success_flag = True

                    to_index=to_index-1

                if tolerance > 0:
                    from_index=from_index+1
                    tolerance -= 1
                else:
                    break

            assert base_from_index not in find_subwords_cache
            if not success_flag and base_from_index + tolerance >= end_index:
                find_subwords_cache[base_from_index] = (True, [], 0, 0)
                return True, [], 0, 0

            find_subwords_cache[base_from_index] = (success_flag, best_tokenization, max_len, best_score)
            return success_flag, best_tokenization, max_len, best_score

        success_flag = False
        tokenization = []

        #big_word = "putyourtesthashtagstringheretotestthisfunction"

        #print("[Hashtag Tokenizer] Tokenizing %s" % big_word)
        maximum_len = max(20, len(big_word)/2)
        maximum_tolerance = 5 if len(big_word) < 15 else int(len(big_word)/3)

        for tolerance in range(0, maximum_tolerance):
            success_flag, tokenization, max_tokenization_len, tokenization_score = find_subwords(0, len(big_word), maximum_len, tolerance)
            if success_flag:
                break

        # if len(big_word) > 50:
        #     if not success_flag:
        #         print("[Hashtag Tokenizer] Failed  (max_len = %d) for:\t %s " % (maximum_len, word))
        #     else:
        #         print("[Hashtag Tokenizer] Success (max_len = %d) for:      %s\n"
        #               "                                                --> [%s]" % (maximum_len, word, ', '.join(tokenization)))

        #if not success_flag:
        #    print("[Hashtag Tokenizer] Failed  (max_len = %d) for:\t %s " % (maximum_len, word))


        if success_flag:
            res_tokenization=[TextRegularizer.tags['hashtag_begin']];
            res_tokenization+=tokenization;
            res_tokenization.append(TextRegularizer.tags['hashtag_end'])
            return True, res_tokenization
        else:
            return False, None
    else:
        return False, None


def _tag_number(word, word_to_occurrence):
    number_of_digits = sum(c.isdigit() for c in word)
    if number_of_digits>3 and len(word)<=number_of_digits+3 and number_of_digits not in word_to_occurrence:
        return True, [TextRegularizer.tags['number']]   # replace number by tag
    else:
        return False, None

def bind_vocabulary(regularizing_func, vocabulary):
    def bind_regularizing_func(*args,**kwargs):
        return regularizing_func(*args,word_to_occurrence=vocabulary.word_to_occurrence,**kwargs)
    return bind_regularizing_func


class TextRegularizer:
    static_regularizing_functions = [_convert_sad_emoticons,
                                     _convert_happy_emoticons,
                                     _convert_haha,
                                     _convert_happy_birthday]

    vocab_regularizing_functions = [_tag_number,
                                    _convert_hashtag]

    def __init__(self, final_vocabulary, internal_vocabulary):
        bound_regularizing_functions=[(bind_vocabulary(regularizing_func, final_vocabulary)
                                       if regularizing_func != _convert_hashtag
                                       else bind_vocabulary(regularizing_func, internal_vocabulary))
                                       for regularizing_func in TextRegularizer.vocab_regularizing_functions ]

        self.regularizing_functions = TextRegularizer.static_regularizing_functions + bound_regularizing_functions

    def regularize_word(self, word):
        for regularizing_func in self.regularizing_functions:
            matched, new_word=regularizing_func(word)
            if matched:
                return new_word
        return [word]

    def regularize_word_static(self, word):
        for regularizing_func in TextRegularizer.static_regularizing_functions:
            matched, new_word=regularizing_func(word)
            if matched:
                return new_word
        return [word]

    tags = { 'sad_emoticon':   '<:(>',
             'happy_emoticon': '<:)>',
             'haha':           '<haha>',
             'happybirthday':  '<happybirthday>',
             'time':           '<time>',
             'date':           '<date>',
             'number':         '<number>',
             'hashtag_begin':  '<hashtag_begin>',
             'hashtag_end':    '<hashtag_end>'}

    @classmethod
    def get_special_words(cls):
        '''
        Used to extend dictionary with replacement symbols that will be generated by regularizer
        :return: list of replacement symbols
        '''
        return list(TextRegularizer.tags.values()) # using "<...>" to denote replacement symbols/tags except for smileys (clear)
