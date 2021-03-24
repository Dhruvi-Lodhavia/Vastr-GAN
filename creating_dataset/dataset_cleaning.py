import os  

def should_remove_line(line, stop_words):
    return any([word in line for word in stop_words])
len = 400
for i in range(len):
    stop_words = [ "Features:","returned","Disclaimer:","Special Technique/Craft:", "Craft:","About the Brand","HEIQ VIROBLOCK","Features","Disclaimer","PRODUCT DETAILS","ORIGIN","origin","Origin:","Blouse Piece"]
    stop_words1 =["Design Details","Design Detail","blouse piece"]
    #text/kurta_men/kurta_men
    file = "text/Anarkali_women/Anarkali_women"
    filename = file +str(i)+".txt"
    
    with open(filename,"r") as f, open(file + "_"+str(i)+".txt", "w") as working:    
        for line in f:   
            if not should_remove_line(line, stop_words):  
                working.write(line)  
            else:
                break
    with open(filename,"w") as f, open(file+"_"+str(i)+".txt","r") as working:   
        for line in working:
            if not should_remove_line(line, stop_words1):  
                f.write(line) 
    os.remove(file+"_"+str(i)+".txt")

        