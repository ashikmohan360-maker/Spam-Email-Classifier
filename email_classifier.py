from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "win money now",
    "free prize claim now",
    "limited offer buy now",
    "you won a lottery",
    "claim your free reward",
    "meeting at 10 am",
    "project discussion tomorrow",
    "lunch with team",
    "schedule a call",
    "weekly report attached"
]

labels = [1,1,1,1,1,0,0,0,0,0]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

test_email = ["Congratulations! You won a free gift"]
test_vec = vectorizer.transform(test_email)

prediction = model.predict(test_vec)[0]
print("Spam" if prediction == 1 else "Not Spam")
