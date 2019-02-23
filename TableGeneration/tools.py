


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from PIL import Image
from io import BytesIO

globalminx=0
globalminy=0
globalmaxx=0
globalmaxy=0


def init_global_mins(startloc,endloc):
    global globalminx,globalmaxx,globalminy,globalmaxy
    globalminx = startloc['x']
    globalminy = startloc['y']
    globalmaxx = endloc['x']
    globalmaxy = endloc['y']

def update_global_x_y(xmin,ymin,xmax,ymax):
    global globalminx,globalminy,globalmaxx,globalmaxy
    if(xmin<globalminx):
        globalminx=xmin
    if(ymin<globalminy):
        globalminy=ymin

    if(xmax>globalmaxx):
        globalmaxx=xmax
    if(ymax>globalmaxy):
        globalmaxy=ymax



def html_to_img(htmlpath,outimgpath,id_count):
    opts = Options()
    opts.set_headless()
    assert opts.headless
    driver = Firefox(options=opts)
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
        update_global_x_y(xmin, ymin, xmax, ymax)
        bboxes.append([lentext,txt,xmin,ymin,xmax,ymax])
        # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)

    png = driver.get_screenshot_as_png()
    driver.stop_client()
    driver.quit()
    im = Image.open(BytesIO(png))
    im = im.crop((globalminx - 5, globalminy - 5, globalmaxx + 10, globalmaxy + 5))
    im.save(outimgpath,dpi=(600,600))
    return bboxes

