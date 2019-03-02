


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS

from selenium.webdriver.firefox.options import Options
from PIL import Image
from io import BytesIO

globalmaxx=0
globalmaxy=0


def init_global_mins(startloc,endloc):
    global globalmaxx,globalmaxy
    globalmaxx = endloc['x']
    globalmaxy = endloc['y']

def update_global_x_y(xmax,ymax):
    global globalmaxx,globalmaxy

    if(xmax>globalmaxx):
        globalmaxx=xmax
    if(ymax>globalmaxy):
        globalmaxy=ymax


def html_to_img(htmlpath,outimgpath,id_count):
    global globalmaxy,globalmaxx
    opts = Options()
    opts.set_headless()
    assert opts.headless
    driver=PhantomJS()
    #driver = Firefox(options=opts)
    driver.get(htmlpath)

    element = WebDriverWait(driver, 500).until(EC.presence_of_element_located((By.ID, '1')))

    WebDriverWait(driver, 500).until(EC.visibility_of(element))
    startloc = driver.find_element_by_id('start').location
    endloc = driver.find_element_by_id('end').location
    init_global_mins(startloc,endloc)

    bboxes=[]
    for id in range(1, id_count):
        e = driver.find_element_by_id(str(id))
        txt=e.text.strip()
        lentext=len(txt)
        loc = e.location
        size_ = e.size
        xmin = loc['x']
        ymin = loc['y']
        xmax = int(size_['width'] + xmin)
        ymax = int(size_['height'] + ymin)
        update_global_x_y(xmax, ymax)
        bboxes.append([lentext,txt,xmin,ymin,xmax,ymax])
        # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)

    png = driver.get_screenshot_as_png()
    driver.stop_client()
    driver.quit()
    im = Image.open(BytesIO(png))

    width,height=im.size

    im = im.crop((0,0, width, 768))
    #print('\nImage shape:',im.size)
    im.save(outimgpath,dpi=(600,600))
    return bboxes

