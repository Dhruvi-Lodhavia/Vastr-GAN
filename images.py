
from PIL import Image
length = [148,107,221,94,93,69,273,90,186,82,105,175,183]
class_name = ['Anarkali_women','banarasi_sarees_women','casual_Shirt_men','casual_Shirt_women','formal_Shirt_men','formal_Shirt_women','printed_kurtas_men','printed_kurtis_women','round_neck_tshirt_men','round_neck_tshirt_women','Sherwani_men','solid_straight_kurtas_men','solid_straight_kurti']
f = open("images.txt","w")
f1 = open("classes.txt","w")
f2 = open("image_class_labels.txt","w")
f3 = open("bounding_boxes.txt","w")
i = 1
image_num = 1
class_label = 1
for i in range(len(length)):
    file1 = str(i)+" "+str(class_name[i])
    f1.write(file1)
    f1.write("\n")
    for j in range(0,length[i]):
        f2.write(str(image_num)+str(" ")+str(class_label))
        f2.write("\n")
        file = str(image_num)+" "+str(class_name[i])+"/"+str(class_name[i])+ str(j)+"jpg"
        f.write(file)
        f.write("\n")

        image_name = "images/"+str(class_name[i]+"/"+str(class_name[i])+str(j)+".jpg")
        im = Image.open(image_name)
        width, height = im.size
        width = float(width)
        height = float(height)
        f3.write(str(image_num)+" "+str(0.0)+" "+str(0.0)+" "+str(width)+" "+str(height))
        f3.write("\n")
        image_num+=1
    class_label+=1
