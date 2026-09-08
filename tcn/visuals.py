"""Export observable raster streams to images and animated GIFs."""
from pathlib import Path
import numpy as np
from PIL import Image

def render_episode(host,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);streams={};paths=[]
    for i,record in enumerate(host.records):
        for name,timed in record.observations.items():
            if 'pixels' not in name and name!='raster':continue
            value=timed.value.decoded
            if not isinstance(value,tuple) or len(value)!=4:continue
            h,w,c,data=value;a=np.array(data,dtype=np.uint8).reshape(h,w,c);im=Image.fromarray(a[:,:,0] if c==1 else a)
            key=name.replace('/','_');path=out/f'{key}_{i:04d}.png';im.save(path);paths.append(str(path));streams.setdefault(key,[]).append(im)
    for name,images in streams.items():
        path=out/(name+'.gif');images[0].save(path,save_all=True,append_images=images[1:],duration=100,loop=0);paths.append(str(path))
    return paths
