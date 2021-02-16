from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


import urllib.request
import os
import json
import time
driver = webdriver.Chrome(ChromeDriverManager().install())
# driver = webdriver.Chrome('chromedriver')
driver.get('https://www.myntra.com/')
time.sleep(5)
search_string = "round neck tshirt women"

driver.find_element_by_class_name('desktop-searchBar').send_keys(search_string)
driver.find_element_by_class_name('desktop-submit').click()
links = []
i = 1

image_limit = 0

while(1):  
  
    for product_base in driver.find_elements_by_class_name('product-base'):
        print(image_limit,"here") 
        l = driver.find_element_by_xpath('//*[@id="desktopSearchResults"]/div[2]/section/ul/li[{}]/a'.format(i)).get_attribute('href')   
        i=i+1
        links.append(l)
        image_limit+=1
    try:
        if(image_limit<= 175):
            print("enter")
            # lim = lim + 50
            #driver.find_element_by_tag_name('//*[@id="desktopSearchResults"]/div[2]/section/div[2]/ul/li[12]').click()
            driver.find_element_by_class_name('pagination-next').click()
            time.sleep(5)  
            
            i = 1
        else:
            break

    except:
        break
        


# print(links)
content = []
count =0
search_string = "round_neck_tshirt_women"
for i in range(len(links)):
    driver.get(links[i])
    time.sleep(2)
    c = driver.find_element_by_class_name('pdp-product-description-content').text
    d = driver.find_element_by_class_name('pdp-name').text
    print(d)
    itr = 1
    image_tags  = driver.find_elements_by_class_name('image-grid-image')[0]
    image_path = os.path.join("images"+"/"+search_string+"/"+search_string+str(count)+".jpg")
    urllib.request.urlretrieve( image_tags.get_attribute('style').split("url(\"")[1].split("\")")[0],image_path)
    
    
    f= open("text/"+search_string+"/"+search_string+str(count)+".txt","w+")
    f.write(d)
    f.write("\n")
    f.write(c)
    f.close()
    count+=1
    # c = driver.find_element_by_class_name('').text
    content.append(c)
