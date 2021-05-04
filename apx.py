from tkinter import *
from classifier import main
from tkinter import filedialog
	
def browseFiles():
		filename = filedialog.askopenfilename(initialdir = "/home/mittooji/Downloads/",
																					title = "Select a File",
																					filetypes = (("Video files","*.mp4*"),
																											 ("all files","*.*")))
		prediction = main(filename)
		label_file_explorer.configure(text="Anomaly Label: "+prediction)
			
window = Tk()
window.title('Anomaly Detection')
window.geometry("750x200")
window.config(background = "white")
label_file_explorer = Label(window,
														text = "Anomaly Detection Project",
														width = 100, height = 4,
														fg = "blue")
button_explore = Button(window,
												text = "Select Files",
												command = browseFiles)
	
button_exit = Button(window,
										 text = "Close App",
										 command = exit)
label_file_explorer.grid(column = 1, row = 1)
button_explore.grid(column = 1, row = 2)
button_exit.grid(column = 1,row = 3)
window.mainloop()