# TextEmbedding
Cleaning and creating text embeddings


Issues trying to address:

1) What to do with words with wrong spellings that doesnt have a word in the pre trained embedding
2) What if the pre-trained embedding has an representation for a wrongly spelled word as well as a correctly
3) What if non english words are present either in the text or the embedding - eg. spanish or chinese slang


What we know:
Textblob has the ability to suggest corrections. But some slang words are missing. e.g. fuckk is corrected to luck.
Can we change Textblob dictionary? are there better spell correctors available. its actually quite tough to train a spell corrector. 
If we use wikipedia, there wont be many slangs. If we use twitter or other sources there will be so many slang words, we will have a hard time figuring out whats correct and whats not. 
Maybe go back to the idea of having the emebedding keys defining the dictionary and writing a fucntion to find the closest matching word from the list of keys.  

Todo:
Check novigs spell checker http://norvig.com/spell-correct.html
Or the sligltly modified version and see if we can use them https://github.com/pirate/spellchecker/blob/master/spellchecker.py
OR use the python autocorrect module - simplest




Lets make some assumptions:
1) Mostly its english. But some entire sentences can be some other language. we can do language detection and translation using TextBlob
2) 