import os
class Bank:
    def __init__(self):
        self.Client_detail_list = []
        self.Logged_in=False
        self.Amount=0
        self.Transfer_amount=False
    def register(self,name,ph_no,password):
        Amount=self.Amount
        conditions=True
        if len(str(ph_no))>10 or len(str(ph_no))<10:
            print("Invalid Phone Number!! check your phone number")
            conditions=False
        SpecialSym = ['$', '@', '#', '%']
        if len(password) < 8:
            print('length should be at least 8')
            conditions = False

        if len(password) > 15:
            print('length should be not be greater than 15')
            conditions= False

        if not any(char.isdigit() for char in password):
            print('Password should have at least one numeral')
            conditions = False

        if not any(char.isupper() for char in password):
            print('Password should have at least one uppercase letter')
            conditions= False

        if not any(char.islower() for char in password):
            print('Password should have at least one lowercase letter')
            conditions = False

        if not any(char in SpecialSym for char in password):
            print('Password should have at least one of the symbols $@#%')
            conditions = False
        if conditions==True:
            print("Account created Successfully")
            self.Client_detail_list = [name,ph_no,password,Amount]
            with open(f"{name}.txt","w") as f:
                for details in self.Client_detail_list:
                    f.write(str(details)+"\n")

    def login(self,name,ph_no,password):
        with open(f"{name}.txt","r") as f:
            details=f.read()
            self.Client_detail_list=details.split("\n")
            if str(ph_no) in str(self.Client_detail_list):
                if str(password) in str(self.Client_detail_list):
                    self.Logged_in=True
                    print(f"{name} logged in ")
                    self.Amount=int(self.Client_detail_list[3])
                    self.name=name
                else:
                    print("Wrong Password!")
            else:
                print("wrong details!")

    def add_amount(self,amount):
        if amount>0:
            self.Amount+=amount
            with open(f"{name}.txt", "r") as f:
                details = f.read()
                self.Client_detail_list = details.split("\n")
            with open(f"{name}.txt", "w") as f:
                f.write(details.replace(str(self.Client_detail_list[3]),str(self.Amount)))
                print("Amount added Successfully!!")
        else:
            print("Please enter correct value of Amount")

    def Withdraw_amount(self, amount):
        if amount > 0:
            self.Amount -= amount
            with open(f"{name}.txt", "r") as f:
                details = f.read()
                self.Client_detail_list = details.split("\n")
            with open(f"{name}.txt", "w") as f:
                f.write(details.replace(str(self.Client_detail_list[3]), str(self.Amount)))
                print("Amount withdrawn Successfully")

    def Transfer(self,amount,name,ph_no):
        if amount >0 and amount<=self.Amount:
            with open(f"{name}.txt","r") as f:
                details=f.read()
                self.Client_detail_list=details.split("\n")
                if str(ph_no) in self.Client_detail_list:
                    self.Transfer_amount=True
                else:
                    print("Wrong Phone number")
            if self.Transfer_amount==True:
                with open(f"{name}.txt","w") as f:
                    new_balance=int(self.Client_detail_list[3])+amount
                    left_balance=int(self.Amount)-amount
                    f.write(details.replace(str(self.Client_detail_list[3]), str(new_balance)))
                with open(f"{self.name}.txt","r") as f:
                    details2=f.read()
                    self.Client_detail_list=details2.split("\n")
                with open(f"{self.name}.txt", "w") as f:
                    f.write(details2.replace(str(self.Client_detail_list[3]), str(left_balance)))
                print("Transferred Successfully")
        elif amount>self.Amount:
            print("Not Enough Balance")
        elif amount<0:
            print("Enter correct amount")
        else:
            print("Something went wrong😢")
    def Balance(self):
        print("Current Balance : ",self.Amount)

    def profile(self):
        with open(f"{name}.txt", "r") as f:
            details = f.read()
            self.Client_detail_list = details.split("\n")
            print("Name :",self.Client_detail_list[0])
            print("Phone Number :",self.Client_detail_list[1])
            print("Bank Balance :",self.Client_detail_list[3])
            print("\n----------------------------------------\n")

            print("Do yo want to edit profile  ")
            choice=input("y/n\n")
            while True:
                if choice=="y" or choice=="Y":
                    print("1.Change phone number")
                    print("2.Change password")
                    print("3.Back to menu")
                    option=int(input())
                    if option==1:
                        print("New Phone Number : ")
                        phone_no=int(input())
                        if len(str(phone_no)) > 10 or len(str(phone_no)) < 10:
                            print("Invalid Phone Number!! check your phone number")
                        else:
                            with open(f"{name}.txt", "r") as f:
                                details = f.read()
                                self.Client_detail_list = details.split("\n")
                            with open(f"{name}.txt", "w") as f:
                                f.write(details.replace(str(self.Client_detail_list[1]), str(phone_no)))
                            print("Phone number changed successfully!")



                    elif option==2:
                        print("New Password : ")
                        password = input()
                        conditions=True
                        SpecialSym = ['$', '@', '#', '%']
                        if len(password) < 8:
                            print('length should be at least 8')
                            conditions = False

                        if len(password) > 15:
                            print('length should be not be greater than 15')
                            conditions = False

                        if not any(char.isdigit() for char in password):
                            print('Password should have at least one numeral')
                            conditions = False

                        if not any(char.isupper() for char in password):
                            print('Password should have at least one uppercase letter')
                            conditions = False

                        if not any(char.islower() for char in password):
                            print('Password should have at least one lowercase letter')
                            conditions = False

                        if not any(char in SpecialSym for char in password):
                            print('Password should have at least one of the symbols $@#%')
                            conditions = False
                        if conditions==True:
                            with open(f"{name}.txt", "r") as f:
                                details = f.read()
                                self.Client_detail_list = details.split("\n")
                            with open(f"{name}.txt", "w") as f:
                                f.write(details.replace(str(self.Client_detail_list[2]), str(password)))
                            print("Password changed successfully!")
                    elif option==3:
                        break
                    else:
                        print("Please input correct option ")
                elif choice=="n" or choice=="N":
                    break
                else:
                    print("Please enter y or n!!!")
                    break

if __name__=="__main__":
    b1=Bank()
    print("Welcome to FA Bank")
    print("1.Login")
    print("2.Create a new Account")
    user=int(input("Enter Your Choice :\n"))

    if user==1:
        print("Logging in")
        name = input("Enter Name : ")
        ph_no = int(input("Enter Phone Number : "))
        password = input("Enter Password : ")
        b1.login(name,ph_no,password)
        while True:
            if b1.Logged_in:
                print("\n-----------------------\n")
                print("1.Add Amount")
                print("2.Withdraw Amount")
                print("3.Transfer Amount")
                print("4.Bank Balance")
                print("5.My Profile")
                print("6.Logout")
                login_user=int(input("\n"))
                if login_user==1:
                    amount=int(input("Enter amount to add:-\n"))
                    b1.add_amount(amount)

                elif login_user==2:
                    amount = int(input("Enter amount to withdraw:-\n"))
                    b1.Withdraw_amount(amount)

                elif login_user==3:
                    name=input("Whom to Transfer ?\n")
                    ph_no=int(input("Phone number : \n"))
                    amount=int(input("Amount : \n"))
                    b1.Transfer(amount,name,ph_no)

                elif login_user==4:
                    b1.Balance()

                elif login_user==5:
                    b1.profile()

                elif login_user==6:
                    break

                else:
                    print("Please enter correct option")

    elif user==2:
        os.system('clear')
        print("Creating a new Account")
        name=input("Enter Name : ")
        ph_no=int(input("Enter Phone Number : "))
        password = input("Enter Password : ")
        b1.register(name,ph_no,password)
    else:
        print("Please Enter Correct Choice!")



