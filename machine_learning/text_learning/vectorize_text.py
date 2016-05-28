
# coding: utf-8

# ## Text Learning Mini-Project
# 
# In the beginning of this class, you identified emails by their authors using a number of supervised classification algorithms. In those projects, we handled the preprocessing for you, transforming the input emails into a TfIdf so they could be fed into the algorithms. Now you will construct your own version of that preprocessing step, so that you are going directly from raw data to processed features.
# 
# You will be given two text files: one contains the locations of all the emails from Sara, the other has emails from Chris. You will also have access to the parseOutText() function, which accepts an opened email as an argument and returns a string containing all the (stemmed) words in the email.
# 
# 

# ## Warming Up with parseOutText()
# 
# You’ll start with a warmup exercise to get acquainted with parseOutText(). Go to the tools directory and run parse_out_email_text.py, which contains parseOutText() and a test email to run this function over.
# 
# parseOutText() takes the opened email and returns only the text part, stripping away any metadata that may occur at the beginning of the email, so what's left is the text of the message. We currently have this script set up so that it will print the text of the email to the screen, what is the text that you get when you run parseOutText()?

# In[1]:


from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        




    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()


# ## Deploying Stemming
# In parseOutText(), comment out the following line: 
# 
# words = text_string 
# 
# Augment parseOutText() so that the string it returns has all the words stemmed using a SnowballStemmer (use the nltk package, some examples that I found helpful can be found here: http://www.nltk.org/howto/stem.html ). Rerun parse_out_email_text.py, which will use your updated parseOutText() function--what’s your output now?
# 
# Hint: you'll need to break the string down into individual words, stem each word, then recombine all the words into one string.

# In[ ]:


from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
  """ given an opened email file f, parse out all text below the
      metadata block at the top
      (in Part 2, you will also add stemming capabilities)
      and return a string that contains all the words
      in the email (space-separated) 
      
      example use case:
      f = open("email_file_name.txt", "r")
      text = parseOutText(f)
      
      """


  f.seek(0)  ### go back to beginning of file (annoying)
  all_text = f.read()
  bag_of_words =''
  

  ### split off metadata
  content = all_text.split("X-FileName:")
  words = ""
  if len(content) > 1:
      ### remove punctuation
      text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

      ### project part 2: comment out the line below
      ## words = text_string
      words = text_string.split()

      ### split the text string into individual words, stem each word,
      stemmer = SnowballStemmer("english")
       
      bag_of_words = [stemmer.stem(word) for word in words]
      ### and append the stemmed word to words (make sure there's a single
      ### space between each stemmed word)
      bag_of_words = (' '.join(bag_of_words)) 
      
  return bag_of_words

  

def main():
  ff = open("../text_learning/test_email.txt", "r")
  text = parseOutText(ff)
  print text



if __name__ == '__main__':
  main()


# ## Clean Away "Signature Words"
# In vectorize_text.py, you will iterate through all the emails from Chris and from Sara. For each email, feed the opened email to parseOutText() and return the stemmed text string. Then do two things:
# 
# remove signature words (“sara”, “shackleton”, “chris”, “germani”--bonus points if you can figure out why it's "germani" and not "germany")
# append the updated text string to word_data -- if the email is from Sara, append 0 (zero) to from_data, or append a 1 if Chris wrote the email.
# Once this step is complete, you should have two lists: one contains the stemmed text of each email, and the second should contain the labels that encode (via a 0 or 1) who the author of that email is.
# 
# Running over all the emails can take a little while (5 minutes or more), so we've added a temp_counter to cut things off after the first 200 emails. Of course, once everything is working, you'd want to run over the full dataset.
# 
# In the box below, put the string that you get for word_data[152].
# 

# 
# 

# In[ ]:


import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0



for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        
        
        #temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            #print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened 
            
            text =  parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            word_data.append(text.replace("sara",'').replace("shackleton",'').replace("chris",'').replace("germani",'').replace("sshacklensf",'').replace("sshacklmsncom",''))
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name =="sara" :
                from_data.append(0)
            else :
                from_data.append(1)
            
   
             

           

           


            email.close()

print "emails processed"
print word_data[152]
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )




# ## Transform the word_data into a tf-idf matrix using the sklearn TfIdf transformation. Remove english stopwords.
# 
# You can access the mapping between words and feature numbers using get_feature_names(), which returns a list of all the words in the vocabulary. How many different words are there?

# In[ ]:




# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
 

vectorizer = TfidfVectorizer (stop_words="english")
vectorizer.fit_transform(word_data)
print "#feature counts : ", len(vectorizer.get_feature_names())
print vectorizer.get_feature_names()[34597]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



