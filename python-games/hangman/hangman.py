#hangman game using python
import random
import string


from words import words

def get_valid_word(words):
    word= random.choice(words)  #randomly chooses something from the list of words
    while '-' in word or " " in word:
        word = random.choice(words)

    return word.upper()

def hangman():
    word=get_valid_word(words)
    word_letters = set(word)    #letters in the word
    alphabet = set(string.ascii_uppercase)
    used_letters= set() #what user has guessed

    lives=6

    #getting user input
    while len(word_letters) > 0 and lives > 0: 
        #letters used
        print('You have', lives, 'lives left and you have used these letters: ', ' '.join(used_letters))
        
        #what current word is
        word_list= [letter if letter in used_letters else '_' for letter in word]
        print("Current word: ", ' '.join(word_list))

        user_letter= input('Guess a letter:').upper()
        if user_letter in alphabet - used_letters:
            used_letters.add(user_letter)
            if used_letters in word_letters:
                word_letters.remove(user_letter)
            
            else:
                lives= lives - 1 #takes away life if wrng guess
                print("Letter is not in the word.")
    
        elif user_letter in used_letters:
            print("You have already used that character. PLease try again.")
    
        else:
            print("Invalid character. PLease try again.")

    if lives ==0:
        print("You died, sorry the word was", word)
    else:
        print("You guessed the word", word, '!!')


hangman()