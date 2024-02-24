# Health Management System
# 3 clients - Person1, Person2 and Person3

def getdate():
    import datetime
    return datetime.datetime.now()

# Total 6 files - 3 for food lock and 3 for exercise
# write a function that when executed takes as input client name
# One more function to retrieve exercise or food for any client

import datetime
def gettime():
    return datetime.datetime.now()
def take(k):
    if k==1:
        c=int(input("Enter 1 for excersise and 2 for food"))
        if(c==1):
            value=input("type here\n")
            with open("Faizaan-ex.txt","a") as op:
                op.write(str([str(gettime())])+": "+value+"\n")
            print("successfully written")
        elif(c==2):
            value = input("type here\n")
            with open("Faizaan-food.txt", "a") as op:
                op.write(str([str(gettime())]) + ": " + value + "\n")
            print("successfully written")
    elif(k==2):
        c = int(input("Enter 1 for excersise and 2 for food"))
        if (c==1):
            value = input("type here\n")
            with open("Arzoo-ex.txt", "a") as op:
                op.write(str([str(gettime())]) + ": " + value + "\n")
            print("successfully written")
        elif (c==2):
            value = input("type here\n")
            with open("Arzoo-food.txt", "a") as op:
                op.write(str([str(gettime())]) + ": " + value + "\n")
            print("successfully written")
    elif(k==3):
        c = int(input("Enter 1 for excersise and 2 for food"))
        if (c==1):
            value = input("type here\n")
            with open("Rahul-ex.txt", "a") as op:
                op.write(str([str(gettime())]) + ": " + value + "\n")
            print("successfully written")
        elif (c==2):
            value = input("type here\n")
            with open("Rahul-food.txt", "a") as op:
                op.write(str([str(gettime())]) + ": " + value + "\n")
            print("successfully written")
    else:
        print("Plz enter valid input (1(Faizaan),2(Arzoo),3(Rahul)")
def retrieve(k):
    if k==1:
        c=int(input("Enter 1 for excersise and 2 for food"))
        if(c==1):
            with open("Faizaan-ex.txt") as op:
                for i in op:
                    print(i,end="")
        elif(c==2):
            with open("Faizaan-food.txt") as op:
                for i in op:
                    print(i, end="")
    elif(k==2):
        c = int(input("Enter 1 for excersise and 2 for food"))
        if (c == 1):
            with open("Arzoo-ex.txt") as op:
                for i in op:
                    print(i, end="")
        elif (c == 2):
            with open("Arzoo-food.txt") as op:
                for i in op:
                    print(i, end="")
    elif(k==3):
        c = int(input("Enter 1 for excersise and 2 for food"))
        if (c == 1):
            with open("Rahul-ex.txt") as op:
                for i in op:
                    print(i, end="")
        elif (c == 2):
            with open("Rahul-food.txt") as op:
                for i in op:
                    print(i, end="")
    else:
        print("Plz enter valid input (Faizaan,Arzoo,Rahul)")
print("health management system: ")
a=int(input("Press 1 for log the value and 2 for retrieve "))

if a==1:
    b = int(input("Press 1 for Faizaan 2 for Arzoo 3 for Rahul "))
    take(b)
else:
    b = int(input("Press 1 for Faizaan 2 for Arzoo 3 for Rahul "))
    retrieve(b)