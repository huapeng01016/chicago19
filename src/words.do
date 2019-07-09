cscript

version 16

python:
import nltk   
import requests
from bs4 import BeautifulSoup


url = "https://www.stata.com/new-in-stata/python-integration/"    
html = requests.get(url)  
text = BeautifulSoup(html.text).get_text() 
print(text)

from wordcloud import WordCloud


# Generate a word cloud image
# wordcloud = WordCloud().generate(text)

wordcloud = WordCloud(max_font_size=75, max_words=100, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
# wordcloud = WordCloud(max_font_size=40).generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
plt.savefig("words.png")

end

