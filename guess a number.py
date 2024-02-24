# n = 18
# number_of_guesses = 1
# print("Number of guesses is limited to 9 times")
# while (number_of_guesses<=9):
#     guess_number = int(input("Guess the number:\n"))
#     if guess_number<18:
#         print("You entered less number please input greater number.\n")
#     elif guess_number>18:
#         print("You entered greater number please input smaller number.\n")
#     else:
#         print("You win! 👏👏 ❤❤ 😊😊\n")
#         print(number_of_guesses,"no. of guesses you took to finish.")
#         break
#     print(9-number_of_guesses,"no. of guesses left")
#     number_of_guesses = number_of_guesses + 1
#
# if number_of_guesses>9:
#     print("Game over","You loose 👎👎")


## Another Code

import random

number = random.randrange(0, 50)

number_of_guesses = 1
print("GUESS THE NUMBER\n  ,, HINT: THE NUMBER IS BETWEEN 0 TO 50 ")
print("NUMBER OF GUESSES IS LIMITED TO 5 TIMES.....!!!!\n")
while (number_of_guesses <= 5):
    guess_number = int(input("GUESS THE NUMBER    ?????\n"))
    if guess_number < number:
        print(" ....ENTER SOME GREATER NUMBER\n \n")
    elif guess_number == number:
        print(" YEAHHH!!!  YOU GOT IT RIGHT , YOU WON ! 👏👏")
        print("NUMBER OF GUESSES YOU TOOK TO FINISH THE GAME IS ", number_of_guesses)
        break
    elif guess_number > 50:
        print("GO THROUGH THE HINT !!!\n")

    else:
        print("TRY AGAIN ,,, ENTER SOME SMALLER NUMBER\n")
    print(5 - number_of_guesses, "NO.OF GUESSES LEFT  ")
    number_of_guesses += 1
    if number_of_guesses > 5:
        print("\n \nGAME OVER!!!! YOU ARE A LOSER ")
        print("THE NUMBER WAS ", number)