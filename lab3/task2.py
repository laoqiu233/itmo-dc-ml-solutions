from math import log, exp

# Вбейте данные сюда
spam = 30
not_spam = 10
spam_words = 201
not_spam_words = 38
words = {
    'Coupon': (7, 1),
    'Online': (1, 1),
    'Gift': (4, 1),
    'Access': (12, 5),
    'Money': (3, 6),
    'Investment': (20, 17),
    'Free': (40, 3),
    'Cash': (67, 2),
    'Offer': (20, 1),
    'Membership': (27, 1)
}
text = 'Refund Gift Cash Access Remove Investment Membership'.split(' ')

print('P("спам") =', spam / (spam + not_spam))

f_spam = log(spam / (spam + not_spam))
f_not_spam = log(not_spam / (spam + not_spam))

r = len(list(filter(lambda x: x not in words, text)))

for word in text:
    a, b = words.get(word, (0, 0))
    
    f_spam += log((1 + a) / (spam_words + len(words) + r))
    f_not_spam += log((1 + b) / (not_spam_words + len(words) + r))

print('F("спам") =', f_spam)
print('F("не спам") =', f_not_spam)
print('P("спам"|Письмо) =', exp(f_spam) / (exp(f_spam) + exp(f_not_spam)))