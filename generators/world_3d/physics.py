"""Local procedural MuJoCo construction and exact integration-state snapshots."""
import math
import mujoco
import numpy as np
_DEFAULT_RGBA=(.7,.7,.8,1.)

def _half_extents(shape: str, size, fallback: float = 1.0):
    """Half extents for the geom of ``shape`` given a scalar or [sx,sy,sz] ``size``."""
    if isinstance(size, (list, tuple)) and len(size) == 3:
        sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
        return (sx / 2.0, sy / 2.0, sz / 2.0)
    s = float(size if size is not None else fallback)
    if shape == "sphere":
        r = s / 2.0
        return (r, r, r)
    if shape == "cylinder":
        r = s / 2.0
        return (r, r, s / 2.0)
    return (s / 2.0, s / 2.0, s / 2.0)

def _body_mjcf(bid: str, info: dict) -> str:
    half = info["half"]
    x, y, z = info["x"], info["y"], info["z"]
    rgba = info.get("rgba") or _DEFAULT_RGBA
    rgba_s = " ".join(str(float(c)) for c in rgba)
    mass = info["mass"]
    # use the per-object material (texture + shading) when present, else flat rgba
    has_mat = bool(info.get("texture")) or info.get("specular") is not None
    appear = f"material='mat_{bid}'" if has_mat else f"rgba='{rgba_s}'"
    head = f"<body name='body_{bid}' pos='{x} {y} {z}'><freejoint name='joint_{bid}'/>"
    if info.get("shape") == "mesh" and info.get("mesh"):
        geom = f"<geom name='geom_{bid}' type='mesh' mesh='mesh_{bid}' mass='{mass}' {appear}/>"
        return head + geom + "</body>"
    if info["kind"] == "container":
        # Tray: thin floor + four walls so `inside` is a real containment test.
        hx, hy, hz = half
        t = max(hx, hy) * 0.12
        floor = f"<geom name='geom_{bid}' type='box' size='{hx} {hy} {t}' pos='0 0 {-hz + t}' mass='{mass}' rgba='{rgba_s}'/>"
        wt = hz  # wall half-height
        walls = "".join([
            f"<geom name='geom_{bid}_wn' type='box' size='{hx} {t} {wt}' pos='0 {hy - t} 0' rgba='{rgba_s}'/>",
            f"<geom name='geom_{bid}_ws' type='box' size='{hx} {t} {wt}' pos='0 {-(hy - t)} 0' rgba='{rgba_s}'/>",
            f"<geom name='geom_{bid}_we' type='box' size='{t} {hy} {wt}' pos='{hx - t} 0 0' rgba='{rgba_s}'/>",
            f"<geom name='geom_{bid}_ww' type='box' size='{t} {hy} {wt}' pos='{-(hx - t)} 0 0' rgba='{rgba_s}'/>",
        ])
        return head + floor + walls + "</body>"
    shape = info["shape"]
    if shape == "sphere":
        geom = f"<geom name='geom_{bid}' type='sphere' size='{half[0]}' mass='{mass}' {appear}/>"
    elif shape == "cylinder":
        geom = f"<geom name='geom_{bid}' type='cylinder' size='{half[0]} {half[2]}' mass='{mass}' {appear}/>"
    else:
        geom = f"<geom name='geom_{bid}' type='box' size='{half[0]} {half[1]} {half[2]}' mass='{mass}' {appear}/>"
    return head + geom + "</body>"

P=np.array([[1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])

def build(objects,dimensions=3):
    parts=['<mujoco model="tcn"><compiler angle="radian"/><option timestep="0.004166666666667" gravity="0 0 -9.81" integrator="implicitfast"/><default><geom friction="1 .05 .001"/></default><worldbody><geom name="ground" type="plane" size="30 30 .1"/>']
    motors=[]
    for o in objects:
        pos=P@np.array(o['position']);size=P@np.array(o.get('size',[.5,.5,.5]));shape=o.get('mesh','box')
        info={'x':pos[0],'y':pos[1],'z':pos[2],'half':tuple(size/2),'mass':o.get('mass',1.),'kind':o.get('kind','object'),'shape':shape,'rgba':[c/255 for c in o.get('color',[160,170,190])]+[1.]}
        if o.get('kind')=='door':
            name=o['id']
            parts.append(f'<body name="body_{name}" pos="{pos[0]} {pos[1]} {pos[2]}"><joint name="hinge_{name}" type="hinge" axis="0 0 1" range="0 1.5" damping="2"/><geom type="box" size="{size[0]/2} {size[1]/2} {size[2]/2}" mass="2"/></body>')
            motors.append(f'<position name="motor_{name}" joint="hinge_{name}" kp="40" ctrllimited="true" ctrlrange="0 1.5"/>')
            continue
        if o.get('static') or shape=='plane':
            parts.append(f'<body name="body_{o["id"]}" pos="{pos[0]} {pos[1]} {pos[2]}"><geom type="box" size="{size[0]/2} {size[1]/2} {max(.01,size[2]/2)}"/></body>');continue
        body=_body_mjcf(o['id'],info)
        if o.get('kind')=='robot':
            # An articulated, actuated link is part of each robot, not a label.
            name=o['id'];arm=f'<body name="arm_{name}" pos=".2 0 .15"><joint name="hinge_{name}" type="hinge" axis="0 1 0" range="-1.4 1.4" damping="1"/><geom type="capsule" size=".04 .15" pos=".15 0 0" quat=".7071 0 .7071 0" mass=".1"/></body>'
            body=body[:-7]+arm+'</body>';motors.append(f'<motor name="motor_{name}" joint="hinge_{name}" gear="1" ctrllimited="true" ctrlrange="-3 3"/>')
        parts.append(body)
    parts.append('</worldbody><actuator>'+''.join(motors)+'</actuator></mujoco>')
    model=mujoco.MjModel.from_xml_string(''.join(parts));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data

def snapshot(model,data):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    out=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,out,spec)
    return out.tolist()

def advance(objects,saved,dt,forces=None,torques=None,holds=None,dimensions=3):
    model,data=build(objects,dimensions)
    if saved is not None:mujoco.mj_setState(model,data,np.array(saved),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(model,data)
    data.xfrc_applied[:]=0;data.ctrl[:]=0
    for name,force in (forces or {}).items():data.xfrc_applied[model.body('body_'+name).id,:3]=P@np.array(force)
    for name,torque in (torques or {}).items():data.ctrl[model.actuator('motor_'+name).id]=torque
    steps=max(1,math.ceil(dt/(1/240)));model.opt.timestep=dt/steps
    for _ in range(steps):
        for actor,target in (holds or {}).items():
            if target:
                a=model.body('body_'+actor).id;b=model.body('body_'+target).id
                desired=data.xpos[a]+np.array([.35,0,.2]);delta=desired-data.xpos[b]
                data.xfrc_applied[b,:3]=np.clip(100*delta-5*data.cvel[b,3:],-40,40)+np.array([0,0,model.body_mass[b]*9.81])
        mujoco.mj_step(model,data)
        if dimensions==2:
            for obj in objects:
                bid=model.body('body_'+obj['id']).id
                if model.body_jntnum[bid] and model.jnt_type[model.body_jntadr[bid]]==mujoco.mjtJoint.mjJNT_FREE:
                    j=model.body_jntadr[bid];q=model.jnt_qposadr[j];v=model.jnt_dofadr[j]
                    data.qpos[q+1]=0;data.qvel[v+1]=0
    mujoco.mj_forward(model,data)
    # Keep the model's construction poses immutable; update only rendering poses.
    poses={}
    for obj in objects:
        bid=model.body('body_'+obj['id']).id
        poses[obj['id']]={'position':(P@data.xpos[bid]).tolist(),'matrix':(P@data.xmat[bid].reshape(3,3)@P).tolist(),'velocity':(P@data.cvel[bid,3:]).tolist()}
    for obj in objects:
        if obj.get('kind')=='robot':
            bid=model.body('arm_'+obj['id']).id
            center=data.xpos[bid]+data.xmat[bid].reshape(3,3)@np.array([.15,0,0])
            poses['arm_'+obj['id']]={'position':(P@center).tolist(),'matrix':(P@data.xmat[bid].reshape(3,3)@P).tolist(),'velocity':(P@data.cvel[bid,3:]).tolist()}
    contacts=[(int(data.contact[i].geom1),int(data.contact[i].geom2),float(data.contact[i].dist)) for i in range(data.ncon)]
    return snapshot(model,data),poses,contacts
