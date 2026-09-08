"""Procedural meshes, articulated transforms, and a deterministic triangle rasterizer.

This is generator-side ground truth, never agent preprocessing.
"""
import math
import numpy as np

def mesh(kind='box',segments=12):
    if kind=='plane':
        return np.array([[-.5,0,-.5],[.5,0,-.5],[.5,0,.5],[-.5,0,.5]]),np.array([[0,1,2],[0,2,3]])
    if kind=='box':
        v=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
        f=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        return v,f
    if kind not in {'sphere','cylinder'}:raise ValueError('unknown procedural mesh')
    rows=segments//2 if kind=='sphere' else 1
    vertices=[];faces=[]
    for j in range(rows+1):
        y=.5*math.cos(math.pi*j/rows) if kind=='sphere' else j-.5
        r=.5*math.sin(math.pi*j/rows) if kind=='sphere' else .5
        for i in range(segments):
            angle=2*math.pi*i/segments;vertices.append([r*math.cos(angle),y,r*math.sin(angle)])
    for j in range(rows):
        for i in range(segments):
            a=j*segments+i;b=j*segments+(i+1)%segments;c=a+segments;d=b+segments
            faces.extend([[a,c,b],[b,c,d]])
    if kind=='cylinder':
        for j in (0,1):
            center=len(vertices);vertices.append([0,j-.5,0])
            for i in range(segments):faces.append([center,j*segments+i,j*segments+(i+1)%segments])
    return np.asarray(vertices),np.asarray(faces)

def rotation(angles):
    x,y,z=angles;cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def world_mesh(obj,objects=None,chain=()):
    if obj['id'] in chain:raise ValueError('articulation cycle')
    vertices,faces=mesh(obj.get('mesh','box'),obj.get('segments',12))
    # Scale, shear/deformation, rotation, translation are explicit generating ops.
    vertices=vertices*np.asarray(obj.get('size',[1,1,1]))
    vertices[:,0]+=obj.get('shear',0)*vertices[:,1]
    vertices=vertices@np.asarray(obj.get('matrix',rotation(obj.get('rotation',[0,0,0])))).T+obj.get('position',[0,0,0])
    parent=obj.get('parent')
    while parent:
        if parent in chain or parent==obj['id']:raise ValueError('articulation cycle')
        p=next(x for x in objects if x['id']==parent)
        vertices=vertices@rotation(p.get('rotation',[0,0,0])).T+p.get('position',[0,0,0])
        chain=(*chain,parent);parent=p.get('parent')
    return vertices,faces

def render(objects,camera=None,width=48,height=48):
    camera=camera or {'eye':[3,3,5],'target':[0,.5,0]}
    eye=np.array(camera['eye'],float);target=np.array(camera['target'],float)
    forward=target-eye;forward/=max(np.linalg.norm(forward),1e-9)
    right=np.cross(forward,np.array([0.,1.,0.]));right/=max(np.linalg.norm(right),1e-9)
    up=np.cross(right,forward)
    focal=.5*height/math.tan(camera.get('fov',60)*math.pi/360)
    rgb=np.empty((height,width,3),np.uint8);rgb[:]=[24,30,43]
    depth=np.full((height,width),np.inf);labels=np.full((height,width),-1,dtype=np.int32);normals=np.zeros((height,width,3))
    light=np.array([.3,.8,.5]);light/=np.linalg.norm(light)
    for index,obj in enumerate(objects):
        if not obj.get('visible',True):continue
        vertices,faces=world_mesh(obj,objects);rel=vertices-eye
        cam=np.stack((rel@right,rel@up,rel@forward),axis=-1)
        proj=np.stack((width/2+focal*cam[:,0]/np.maximum(cam[:,2],.01),height/2-focal*cam[:,1]/np.maximum(cam[:,2],.01)),axis=-1)
        texture=None
        if obj.get('text') is not None:
            from generators.raster_text.generator import render_text
            texture=np.asarray(render_text(obj['text'],128,64,14))
        for face in faces:
            if np.any(cam[face,2]<=.05):continue
            pts=proj[face];lo=np.maximum(np.floor(pts.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(pts.max(0)).astype(int),[width-1,height-1])
            if np.any(lo>hi):continue
            a,b,c=pts;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-9:continue
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
            w0=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
            w1=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;w2=1-w0-w1
            inv=w0/cam[face[0],2]+w1/cam[face[1],2]+w2/cam[face[2],2]
            z=np.divide(1.,inv,out=np.full_like(inv,np.inf),where=inv>0)
            sl=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1));mask=(w0>=-1e-8)&(w1>=-1e-8)&(w2>=-1e-8)&(z<depth[sl])
            normal=np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]])
            normal/=max(np.linalg.norm(normal),1e-9);shade=.3+.7*abs(float(normal@light))
            colors=np.broadcast_to(np.asarray(obj.get('color',[150,150,170]))*shade,(*z.shape,3)).copy()
            if texture is not None:
                # Texture coordinates are derived from local mesh coordinates.
                base,_=mesh(obj.get('mesh','box'),obj.get('segments',12))
                uv=(base[face][:,[0,2]]+.5)
                u=(w0*uv[0,0]/cam[face[0],2]+w1*uv[1,0]/cam[face[1],2]+w2*uv[2,0]/cam[face[2],2])*z
                v=(w0*uv[0,1]/cam[face[0],2]+w1*uv[1,1]/cam[face[1],2]+w2*uv[2,1]/cam[face[2],2])*z
                tx=np.clip(np.nan_to_num(u)*127,0,127).astype(int);ty=np.clip(np.nan_to_num(v)*63,0,63).astype(int)
                colors=texture[ty,tx]*shade
            rgb[sl][mask]=np.clip(colors[mask],0,255).astype(np.uint8);depth[sl][mask]=z[mask];labels[sl][mask]=index;normals[sl][mask]=normal
    depth[~np.isfinite(depth)]=0
    return rgb,depth,labels,normals
