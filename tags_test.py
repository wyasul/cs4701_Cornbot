import json

train = open('cornbot_train.json')
data = (json.load(train))
traintags = []
for intent in data:
    for tag in data[intent]:
        traintags = traintags + [(tag['tag'])]

validate = open('cornbot_validation.json')
data = (json.load(validate))
validatetags = []
for intent in data:
    for tag in data[intent]:
        validatetags = validatetags + [(tag['tag'])]

test = open('cornbot_test.json')
data = (json.load(test))
testtags = []
for intent in data:
    for tag in data[intent]:
        testtags = testtags + [(tag['tag'])]

difference_validatetest = list(set(testtags) - set(validatetags))
difference_validatetrain = list(set(traintags) - set(validatetags))
difference_testtrain = list(set(traintags) - set(testtags))

print(difference_validatetest, difference_validatetrain, difference_testtrain)
assert len(difference_validatetest) == 0
assert len(difference_validatetrain) == 0
assert len(difference_testtrain) == 0
