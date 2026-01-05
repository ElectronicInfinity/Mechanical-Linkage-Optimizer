import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation
import time
from matplotlib.widgets import Button
start_time = time.time()
GpuMode=False
if GpuMode: #note GPUMODE is untested but may be good.. hard to say
    import cupy as np
else:
    import numpy as np
BullShitFitnessReturn=1000
#parameters
angleincirmentforanimation=0.1
#assert 0 <= removePercent < 100

def get_intersections(x0, y0, r0, x1, y1, r1, Cardinallity):
    #for transparancy this was from some stackoverflow post.. i think, i dont really remeber but now that im reviwing these projects it seems that way
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1
    xdiff=x1-x0
    ydiff=y1-y0
    dsqr=(xdiff)**2 + (ydiff)**2
    d=math.sqrt(dsqr)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        xdiff/=d
        ydiff/=d
        a=(r0**2-r1**2+dsqr)/(2*d)
        h=math.sqrt(r0**2-a**2)

        x2=x0+a*(xdiff)
        y2=y0+a*(ydiff)
        if Cardinallity:
            xOut=x2+h*(ydiff)
            yOut=y2-h*(xdiff)
        else:
            xOut=x2-h*(ydiff)
            yOut=y2+h*(xdiff)
        return np.array([xOut,yOut])

class specimen:
    def __init__(self, Params, TargetList,angulardensity):
        self.Params=Params
        self.TargetList=TargetList
        self.Cardinallity=False

      #  if (Params < 0).any(): TODO THIS MAY NOT BE NESSECAY AND MAY ACTUALLY AUTOMAICALLY ACCOUNT FOR CARDINALLTIES
       #     self.Fitness=BullShitFitnessReturn
         #   
         #   return
        lossL=self.ForwardPass45BarLinkLoss(angulardensity)
        self.Cardinallity=True
        lossR=self.ForwardPass45BarLinkLoss(angulardensity)
        if lossL == None or lossR == None:
            self.Fitness=BullShitFitnessReturn
            return
        else:
            MinLoss=min(lossR,lossL)
            self.Cardinallity = lossR==MinLoss
            self.Fitness=MinLoss
    def ForwardPass45BarLinkLoss(self,angulardensity):
        TwoPi=2*math.pi

        x=self.Params
        Cardinallity=self.Cardinallity
      #  if GpuMode:
      #      pointc=np.array([x[3].get(),0])
     #   else:
        #pointc=np.array([x[3],0])
        sumloss=0
        for target in self.TargetList:
            MinLoss=BullShitFitnessReturn   #thius was otuside target loop im so goofy
            for angle in np.linspace(0,TwoPi,angulardensity):
                pointb=x[0]*np.array([math.cos(angle),math.sin(angle)])
                pointd=get_intersections(*pointb,x[1],x[3],0,x[2],Cardinallity)
                if pointd is None:
                    return None
                pointdiffrenatial=pointd-pointb
                pointe=pointd+pointdiffrenatial*(x[4])/math.sqrt(pointdiffrenatial[0]**2+pointdiffrenatial[1]**2)
                #use np.linalg.norm(diff)?
                differnatial=pointe-target
                MinLoss=min(MinLoss,differnatial[0]**2+differnatial[1]**2)
            sumloss+=MinLoss
        return sumloss
    def DisplayLinkage(self, ShowFullPath=False, SaveAsAGif=False):
        x = self.Params
        Cardinallity = self.Cardinallity
        pointc = np.array([x[3], 0])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        
        # 1. Pre-calculate the path to determine plot limits
        all_pts = []
        path_pts = []
        for AngGle in np.linspace(0, 2 * math.pi, 100):
            pb = x[0] * np.array([math.cos(AngGle), math.sin(AngGle)])
            res = get_intersections(*pb, x[1], *pointc, x[2], Cardinallity)
            if res is not None:
                pd = np.array(res)
                diff = pd - pb
                unit_vec = diff / np.linalg.norm(diff)
                pe = pd + unit_vec * x[4]
                # Collect all joint positions to find the outer bounds
                all_pts.extend([pb, pd, pe, [0, 0], pointc])
                path_pts.append(pe)

        if not all_pts:
            print("Linkage configuration is invalid.")
            return

        # 2. Set static axis limits based on the calculated bounds
        all_pts = np.array(all_pts)
        margin = 1.0  # Add a little breathing room
        ax.set_xlim(np.min(all_pts[:, 0]) - margin, np.max(all_pts[:, 0]) + margin)
        ax.set_ylim(np.min(all_pts[:, 1]) - margin, np.max(all_pts[:, 1]) + margin)

        # 3. Draw static elements (Targets and Full Path)
        for target in self.TargetList:
            ax.plot(*target, 'bo', markersize=8, label='Target' if target is self.TargetList[0] else "")
        
        if ShowFullPath and path_pts:
            px, py = zip(*path_pts)
            ax.plot(px, py, 'r--', alpha=0.4, label="End Effector Path")

        ax.set_title("Optimized 4-Bar Linkage")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # 4. Animation Update Function
        def update(frame):
            angle = angleincirmentforanimation * frame
            
            # Clear previous frame lines but keep limits
            for L in ax.get_lines():
                # We don't want to remove the static path or targets
                if L.get_linestyle() != '--' and L.get_color()!='b':
                    L.remove()

            pb = x[0] * np.array([np.cos(angle), np.sin(angle)])
            result = get_intersections(*pb, x[1], *pointc, x[2], Cardinallity)
            
            if result is None: return 

            pd = np.array(result)
            unit_vec = (pd - pb) / np.linalg.norm(pd - pb)
            pe = pd + unit_vec * x[4]

            # Draw Bars
            ax.plot([0, pb[0]], [0, pb[1]], 'g-', lw=3)               # Crank
            ax.plot([pb[0], pd[0]], [pb[1], pd[1]], 'g-', lw=3)      # Coupler
            ax.plot([pd[0], pointc[0]], [pd[1], pointc[1]], 'g-', lw=3) # Rocker
            ax.plot([pd[0], pe[0]], [pd[1], pe[1]], 'g-', lw=3)      # Extension
            ax.plot([0, pointc[0]], [0, 0], 'g-', alpha=0.3)         # Ground link

            # Draw Joints
            ax.plot([0, pb[0], pd[0], pe[0], pointc[0]], 
                    [0, pb[1], pd[1], pe[1], 0], 'ko', markersize=5)
        
        ani = matplotlib.animation.FuncAnimation(
            fig, update, 
            frames=math.ceil(2*np.pi/angleincirmentforanimation), 
            interval=30
        )
        
        if SaveAsAGif:
            writer = matplotlib.animation.PillowWriter(fps=30)
            ani.save('Linkage_Optimized.gif', writer=writer)
            
        plt.show()

def keyfunc(ob : specimen):
    return ob.Fitness
class EvolutionReturn:
    def __init__(self,BestManList):
        self.List=BestManList
        self.BestMan=sorted(BestManList,key=keyfunc)[0]
        self.LossList=[x.Fitness for x in BestManList]
    def Graph(self,Label=None):
        if Label is None:
            plt.plot(self.LossList)
        else:
            plt.plot(self.LossList,label=Label)

def CompleteEvoltuion(
    targets,
    startingpopulation=None,
    PopualtioNSize=100,
    removePercent=0,
    generationcount=100,
    MutationMultiplier=0.1,
    angulardensity=20,
    startingeccentricty=10,
    SEXUALREPRODUCTIONTEMPTOGGLE=False,
    WeightedEvolution=True,
    SQRTDistributeWeight=False,
    GradientDescentPreStep=False,
    InitalPopulationBoost=1000,
    MixinRandom=None):
    BoosterPopulation=[]
    #booster popualtiuon gives intial creativity
    if InitalPopulationBoost > 0:
        BoosterPopulation=InstnatiateRandomPopulatuion(targets,startingeccentricty,InitalPopulationBoost,angulardensity)
        startingpopulation = sorted(BoosterPopulation,key=keyfunc)[:PopualtioNSize]

    bestManList=[]
    Population=[]
    RemovePopulationCount=int(PopualtioNSize * removePercent / 100) #Bottom howmany are to be removed each iteration
    if startingpopulation is None:
        startingpopulation=InstnatiateRandomPopulatuion(targets,startingeccentricty,PopualtioNSize,angulardensity)
    else:
        Population=startingpopulation
    BestMan : specimen = sorted(Population,key=keyfunc)[0]
    bestManList.append(BestMan)

    for gen in range(generationcount):        
        CreamOfTheCrop=sorted(Population,key=keyfunc)[:len(Population)-RemovePopulationCount] #is this the right way around?
        #Do PreGradiants
        if GradientDescentPreStep:
            NextPop=[]
            for CreamyLinky in CreamOfTheCrop:
                GradientOptimizedLinky=OptimzieWithGradientDescent(CreamyLinky,Epochs=10,h=0.01,lr=0.1)[1]
                NextPop.append(GradientOptimizedLinky)
            CreamOfTheCrop=NextPop
        BabyBuffer=[]
        if WeightedEvolution:
            CreamCropWeightList=[]
            for creamylinkage in CreamOfTheCrop:
                if SQRTDistributeWeight:
                    CreamCropWeightList.append(1/(creamylinkage.Fitness)**(1/5))
                else:
                    CreamCropWeightList.append(1/creamylinkage.Fitness)
            RandomSpecimenList = random.choices(CreamOfTheCrop,k=PopualtioNSize,weights=CreamCropWeightList)
            RandomSpecimenList2 = random.choices(CreamOfTheCrop,k=PopualtioNSize,weights=CreamCropWeightList)
        else:
            print("BUGFIX",CreamOfTheCrop,PopualtioNSize)
            RandomSpecimenList = random.choices(CreamOfTheCrop,k=PopualtioNSize)
            RandomSpecimenList2 = random.choices(CreamOfTheCrop,k=PopualtioNSize)
        MutationNoise = MutationMultiplier*np.random.rand(PopualtioNSize, 5)-MutationMultiplier/2
        i=0
        if not SEXUALREPRODUCTIONTEMPTOGGLE:
            for RandomSpecimen in RandomSpecimenList: #Parralelize this and we gain insane speed
                BabyBuffer.append(specimen(MutationNoise[i]+RandomSpecimen.Params,targets,angulardensity))
                i+=1
        else:
            for RandomSpecimen in RandomSpecimenList: #Parralelize this and we gain insane speed
                BabyBuffer.append(specimen(MutationNoise[i]+(RandomSpecimenList2[i].Params/2)+(RandomSpecimen.Params/2),targets,angulardensity))
                i+=1
        BabyBuffer = sorted(BabyBuffer,key=keyfunc)
        if MixinRandom is not None:
            start = int(PopualtioNSize*(1-MixinRandom))
            replacement = InstnatiateRandomPopulatuion(targets,startingeccentricty,start,angulardensity) # must match the slice length
            BabyBuffer[start:] = replacement
        Population = BabyBuffer
        BestMan : specimen = BabyBuffer[0]
        bestloss=BestMan.Fitness
        print("COMPLETED GENERATION: " + str(gen+1)+"/"+str(generationcount) + " at a loss of: " + str(bestloss))
        bestManList.append(BestMan)
    return EvolutionReturn(bestManList)

Targets=[
        np.array([1,1]),
        np.array([1,0.5]), 
        np.array([-1,0.5]),
        3*np.array([0.13,0.254])
        ]

def InstnatiateRandomPopulatuion(targets,startingeccentricty=5,PopualtioNSize=100,angulardensity=20):
    Population=[]
    Randoms = startingeccentricty*np.random.rand(PopualtioNSize, 5)
    for p in range(PopualtioNSize):
        Population.append(specimen(Randoms[p],targets,angulardensity))
    return Population
def TargetExpeirment(upToNTargets,startingeccentricty=5,PopualtioNSize=100,targets=Targets,angulardensity=20):
    liltargets=[]
    Population=InstnatiateRandomPopulatuion(targets=targets,startingeccentricty=startingeccentricty,PopualtioNSize=PopualtioNSize)
    for n in range(upToNTargets):
        liltargets.append(5*np.array([np.random.rand(),np.random.rand()]))
        CompleteEvoltuion(liltargets,generationcount=50,startingpopulation=Population).Graph(str(n+1)+" Target Points")
def MutationExperiment(upToNTargets):
    for n in range(1,upToNTargets+1):
        CompleteEvoltuion(Targets,MutationMultiplier=0.1*n,generationcount=30).Graph(str(n*0.1)+" MutationMultiplier")
def RemovePercentExperiment(upToNTargets):
     for n in range(1,upToNTargets+1):
        CompleteEvoltuion(targets=AnthonysTargetPoints,
                                PopualtioNSize=100,
                                generationcount=40,
                                SEXUALREPRODUCTIONTEMPTOGGLE=True,
                                startingpopulation=Population,
                                WeightedEvolution=True, removePercent=10*n,SQRTDistributeWeight=True).Graph(str(n*10)+" RemovePercent")

def OptimzieForTargetsWithGuessandCheck(targetpoints,AmountCheck):
    BestMan=None
    LowestLoss=None
    LowestLossList=[]
    Population=InstnatiateRandomPopulatuion(targets=targetpoints,PopualtioNSize=AmountCheck)
    for member in Population:
        loss=member.ForwardPass45BarLinkLoss(20)
        if loss is None:
            LowestLossList.append(LowestLoss)

            continue
        if LowestLoss is None:
            BestMan=member
            LowestLoss=loss
        if LowestLoss>loss:
            BestMan=member
            LowestLoss=loss
        LowestLossList.append(LowestLoss)

    return (LowestLossList,BestMan)
def OptimzieWithGradientDescent(Linkage:specimen,Epochs,h,lr):
    currentlossHistory=[]
    priorLinkage=Linkage
    for epoch in range(Epochs):
        Params=Linkage.Params
        currentloss=Linkage.ForwardPass45BarLinkLoss(20)
        if currentloss is None:
            #how is thispossible..?
            return (currentlossHistory,priorLinkage)
        currentlossHistory.append(currentloss)
        gradients=np.array([0,0,0,0,0])
        for i in range(len(Params)):
            NudgedParameters = Params
            NudgedParameters[i]+=h
            Linkage.Params=NudgedParameters
            NudgedLoss=Linkage.ForwardPass45BarLinkLoss(20)
            if NudgedLoss is None:
                NudgedLoss=currentloss
            gradient=(NudgedLoss-currentloss)/h
            gradients[i]=gradient
        priorLinkage=Linkage
        Linkage.Params=Params-lr*gradients
    return (currentlossHistory,Linkage)
def OptimzieWithGradientDescentMulti(Epochs,h,lr,targetpoints):
    BestMan=None
    LowestLoss=None
    LowestLossList=[]
    Population=InstnatiateRandomPopulatuion(targets=targetpoints,PopualtioNSize=Epochs)
    for member in Population:
        member=OptimzieWithGradientDescent(member,10,h,lr)[1]
        loss=member.ForwardPass45BarLinkLoss(20)
        if loss is None:
            LowestLossList.append(LowestLoss)
            continue
        if LowestLoss is None:
            BestMan=member
            LowestLoss=loss
        if LowestLoss>loss:
            BestMan=member
            LowestLoss=loss
        LowestLossList.append(LowestLoss)
    return (LowestLossList,BestMan)

AnthonysTargetPoints=[
    np.array([1,2]),
    np.array([0.4,1])
]
NewlyDementedTargets=[
    np.array([0.3,0.4]),
            np.array([0.4,0.8]),
            np.array([0.2,0.6]),
            np.array([0.5,-0.1])
]
Population=InstnatiateRandomPopulatuion(targets=NewlyDementedTargets)

class LinkageEditor:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_title("L-Click: Add | R-Click: Remove | Drag: Move \n  0,0 is always forced to be the postion of the root point  \n transforming points while preserving relative postitons still changes solutions")

        
        self.targets = [np.array([5.1, 4])/2,np.array([6, 3])/2]
        self.points_plot, = self.ax.plot([p[0] for p in self.targets], [p[1] for p in self.targets], 'bo', picker=5)
        self.path_plot, = self.ax.plot([], [], 'r--', alpha=0.5)
        self.link_lines = [self.ax.plot([], [], color=c, lw=2)[0] for c in ['red', 'green', 'blue', 'magenta']]
        
        self.dragging_idx = None
        self.ani = None

        # Add Button
        ax_button = plt.axes([0.7, 0.01, 0.2, 0.05])
        self.btn = Button(ax_button, 'Live Optimize')
        self.btn.on_clicked(self.optimize)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        
        plt.show()

    def update_plot(self):
        if self.targets:
            x, y = zip(*self.targets)
            self.points_plot.set_data(x, y)
        else:
            self.points_plot.set_data([], [])
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        
        # Check if clicking existing point
        cont, ind = self.points_plot.contains(event)
        if cont:
            try:
                self.ani.event_source.stop()
            except: 
                print("drag")
            idx = ind['ind'][0]
            if event.button == 3: # Right click remove
                self.targets.pop(idx)
                self.update_plot()
            else: # Left click drag
                self.dragging_idx = idx
        elif event.button == 1: # Left click add
            self.targets.append(np.array([event.xdata, event.ydata]))
            self.update_plot()
        

    def on_drag(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax: return
        self.targets[self.dragging_idx] = np.array([event.xdata, event.ydata])
        self.update_plot()



    def on_release(self, event):
        self.dragging_idx = None
      #  self.ani.event_source.start()

    def optimize(self, event):
        if not self.targets: return
        print("Optimizing...")
        best = CompleteEvoltuion(targets=self.targets,
                                PopualtioNSize=50,
                                generationcount=150,
                                SEXUALREPRODUCTIONTEMPTOGGLE=False,
                                WeightedEvolution=True,
                                removePercent=0,
                                InitalPopulationBoost=1000,#this paramater is very perofmance intsive, i used 1000 but i dont know what hardware you have so i left it lower -  if you can handle its amazing
                                MixinRandom=None)
        self.animate_linkage(best.BestMan)

    def animate_linkage(self, Linkage : specimen):
        if self.ani: self.ani.event_source.stop()
        x = Linkage.Params
        pts = []

        for angle in np.linspace(0, 2*np.pi, 60):
            pb = x[0] * np.array([np.cos(angle), np.sin(angle)])
            pd = get_intersections(*pb, x[1], x[3], 0, x[2], Linkage.Cardinallity)
            if pd is not None:
                pe = pd + (pd - pb) * (x[4] / np.linalg.norm(pd - pb))
                pts.append(pe)
        
        if pts:
            px, py = zip(*pts)
            self.path_plot.set_data(px, py)
        def update(frame):
            angle = frame
            pb = x[0] * np.array([np.cos(angle), np.sin(angle)])
            pd = get_intersections(*pb, x[1], x[3], 0, x[2], Linkage.Cardinallity)
            if pd is None: return self.link_lines
            pe = pd + (pd - pb) * (x[4] / np.linalg.norm(pd - pb))
            
            self.link_lines[0].set_data([0, pb[0]], [0, pb[1]])
            self.link_lines[1].set_data([pb[0], pd[0]], [pb[1], pd[1]])
            self.link_lines[2].set_data([pd[0], x[3]], [pd[1], 0])
            self.link_lines[3].set_data([pd[0], pe[0]], [pd[1], pe[1]])
            return self.link_lines

        self.ani = matplotlib.animation.FuncAnimation(
            self.fig, update, frames=np.linspace(0, 2*np.pi, 60), interval=30, blit=True)
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    LinkageEditor()