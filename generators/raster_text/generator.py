from PIL import Image,ImageDraw,ImageFont
from tcn.generation import Generator,Value,text_value,read_text,image_value,vector_value
TEXT=text_value('',64).type

def render_text(text,width=96,height=32,size=14,foreground=(15,15,15),background=(245,245,245)):
    image=Image.new('RGB',(width,height),background);draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=size)
    draw.text((3,3),text,font=font,fill=foreground)
    return image

class Implementation(Generator):
    action_schema={'wait':{},'write':{'text':TEXT}}
    def initialize(self,address,configuration):
        rng=address.rng();text=configuration.get('text',''.join(rng.choice('abcdefghijk 0123456789') for _ in range(rng.randrange(3,10))))
        return {'time':0.,'tick':0,'text':text,'width':configuration.get('width',64),'height':configuration.get('height',24),'size':rng.randrange(10,17),'response':'','horizon':configuration.get('horizon',4)}
    def advance(self,s,actions,dt,rng):
        for a in actions:
            if a.verb=='write':s['response']=read_text(dict(a.arguments)['text'])
        s['done']=s['tick']+1>=s['horizon'];return s,{}, {'correct':float(s['response']==s['text'])}
    def observe(self,s):
        img=render_text(s['text'],s['width'],s['height'],s['size'])
        return {'pixels':image_value(img)},{'layout':vector_value([3,3,s['size']])},{'text':text_value(s['text'],64),'raster':image_value(img)},{'agent_0':('wait','write')}
