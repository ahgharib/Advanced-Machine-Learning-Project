from tkinter import *
from tkinter import messagebox
#from ann import open_ann_window
#from dt import open_dt_window
#from svm import open_svm_window
from SVM_GUI import open_svm_window
from DTC_GUI import open_dtc_window
from ANN_GUI import open_ann_window
from DTR_GUI import open_dtr_window


ann_features = 4
svm_features = 11
dt_features = 11

root = Tk()


root.title("Advanced Machine Learning project") #for the program title
root.geometry("800x800") #default size
root.minsize(400,400) #user will not be able to resize the window smaller than this
root.maxsize(1000,1000) #user will not be able to resize the window larger than this
root.iconbitmap("/logo.ico")
root.configure(bg="pink")



label = Label(root, text="WELCOME TO OUR AML PROJECT", font=("Arial",20,"bold"), width=35, height=1, bg="lightblue", foreground="purple")
label.pack(pady=5) # PACK(SIDE = LEFT) RIGHT TOP BOTTOM


label = Label(root, text="Choose a Model", font=("Arial",20,"bold"), width=35, height=1, bg="lightblue", foreground="purple")
label.pack(pady=5) # PACK(SIDE = LEFT) RIGHT TOP BOTTOM


options = ["ANN", "DT", "SVM", "DTR"]
selected_option = StringVar()
option_menu = OptionMenu(root, selected_option, *options)
selected_option.set(options[0])
option_menu.configure(bg="yellow", fg="green", font=("Arial",15,"bold"), width=10)
option_menu["menu"].config(bg="yellow", fg="green", font=("Arial",15,"bold"))
option_menu.pack()


def get():
    user_option = selected_option.get()
    if user_option == "ANN":
        messagebox.showinfo("choise selected","Artifitial Neural Network sucssesfully selected")
        open_ann_window()
    elif user_option == "DT":
        messagebox.showinfo("choise selected","Decision Trees sucssesfully selected")
        open_dtc_window()
    elif user_option == "SVM":
        messagebox.showinfo("choise selected","Support Vector Machine sucssesfully selected")
        open_svm_window()
    elif user_option == "DTR":
        messagebox.showinfo("choise selected","Decision Trees Regressor sucssesfully selected")
        open_dtr_window()


my_button = Button(text="Choose",font=("Arial", 15), width=20, bg="red", command=get, borderwidth=3,activebackground="blue") 
my_button.pack(pady=10)
#                                                                                //BUTTON SETTING\\

root.mainloop()