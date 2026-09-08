from generators.world_3d.generator import Implementation as World
class Implementation(World):
    def initialize(self,address,configuration):
        defaults={'agents':2,'objects':[
            {'id':'paper','kind':'paper','mesh':'plane','static':True,'position':[.4,.4,-.4],'size':[.7,.02,.5],'text':'7','color':[240,240,220]},
            {'id':'computer','kind':'computer','mesh':'box','static':True,'position':[-.5,.45,-.5],'size':[.6,.5,.1],'color':[40,50,65]},
            {'id':'container','kind':'container','mesh':'box','position':[1.,.3,-1.],'size':[.6,.5,.6],'color':[130,80,40]},
            {'id':'door','kind':'door','mesh':'box','static':True,'position':[2.,1.,-2.],'size':[.1,2.,1.],'locked':False,'color':[110,70,40]},
            {'id':'button','kind':'button','mesh':'box','static':True,'position':[.8,.65,-.5],'size':[.15,.15,.08],'controls':'door','color':[220,50,40]},
            {'id':'tool','kind':'object','mesh':'cylinder','position':[.3,.6,-.8],'size':[.1,.4,.1],'color':[160,170,180]}
        ]}
        return super().initialize(address,defaults|configuration)
