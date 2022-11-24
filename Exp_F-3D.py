#!/usr/bin/env python
# coding: utf-8

# # Basic python imports and model settings

# In[ ]:


# import underworld.visualisation as vis

from underworld import function as fn
import underworld as uw
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import math
import os
import sys

import time

LoadFromFile = False

resX = 240
resY = 48
resZ = 240

#resX = 128
#resY = 16
#resZ = 128

#resX = 28
#resY = 5
#resZ = 28

H0 = 1000.                 # surface height (snow)
sigma = 10. * H0            # width gaussian bump 10.*H0
a0 = 100.                  # amplitude gaussian bump
alpha = 3. * np.pi / 180.  # dip of surface

minX, maxX = -(100. * H0) / 2., (100. * H0) / 2.
#minX, maxX = -(20. * H0) / 2., (20. * H0) / 2.
minY, maxY = 0., 1100.
minZ, maxZ = -(100. * H0) / 2., (100. * H0) / 2.

g = 9.81
ice_density = 910.

A = 2.140373e-7
n = 1.

print (f'Hello from process {rank} out of {size}')

# generate output path
outputPath = os.path.join(os.path.abspath("."), "PeriZPICoutput_" + str(maxY)+ "_res_" + str(resX) + "_x_" + str(resY) + "_x_" + str(resZ) + "/")
inputPath = outputPath

if rank == 0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
#         os.chdir(outputPath)
comm.Barrier()

print (f'Hello from process {rank} out of {size}')

cell_height = maxY / resY
cell_width = maxX / resX
cell_depth = maxZ / resZ

coord = fn.input()

if(LoadFromFile == True):
    step = 220
    step_out = 10
    nsteps = 10000
    timestep = float(np.load(outputPath+"time"+str(step).zfill(4)+".npy"))

else:
    step = 0
    step_out = 20
    nsteps = 10000
    timestep = 0. 


elementType = "Q1/dQ0"

mesh = uw.mesh.FeMesh_Cartesian(elementType=(elementType),
                                elementRes=(resX, resY, resZ),
                                minCoord=(minX, minY, minZ),
                                maxCoord=(maxX, maxY, maxZ),
                                periodic=[True, False, True])

def mesh_deform(section,fixPoint,mi,case):
    # fixPoint: the position to be refined
    # mi: representing the gradient of mesh resolution; the larger mi, the larger gradient
    if case == 0:
        maxCoord = maxX
        minCoord = minX
    elif case ==1:
        maxCoord = maxY
        minCoord = minY
    else:
        maxCoord = maxZ
        minCoord = minZ  
        
    for index in range(len(section)):

        if  section[index]<=fixPoint:
            zz_sqrt = (fixPoint-section[index])**mi
            zz_sqrt_max = (fixPoint-minCoord)**mi
            section[index] = fixPoint-(fixPoint-section[index])*zz_sqrt/zz_sqrt_max

        if  section[index]>=fixPoint:
            zz_sqrt = (section[index]-fixPoint)**mi
            zz_sqrt_max = (maxCoord-fixPoint)**mi
            section[index] =fixPoint+(section[index]-fixPoint)*zz_sqrt/zz_sqrt_max 
            
    return section

def mesh_deform_Ind(section,Point0,fixPoint,mi,case):
    section_copy = np.copy(section)
#     section[fixPoint_index] = fixPoint
    
    if case == 0:
        maxCoord = maxX
        minCoord = minX
    elif case ==1:
        maxCoord = maxY
        minCoord = minY
    else:
        maxCoord = maxZ
        minCoord = minZ  
        

    # fixPoint: the position to be refined
    # mi: representing the gradient of mesh resolution; the larger mi, the larger gradient
    for index in range(len(section)):

        if  section_copy[index] >= Point0:

            section[index] = fixPoint + (maxCoord-fixPoint)*((section_copy[index]-Point0)/(maxCoord-Point0))**mi
            
    return section


mesh.reset()

dx = (maxX-minX)/resX
dy = (maxZ-minZ)/resZ

with mesh.deform_mesh():
    for index_j in range(resZ+1):
        for index in range(resX+1):              
            start_x = minX+dx*index
            start_y = minZ+index_j*dy
            interface_z =  a0 * np.exp( -(start_x**2. + start_y**2.) / sigma**2.)

            indx = np.where(abs(mesh.data[:,0]-start_x)<0.01*dx)
            indy = np.where(abs(mesh.data[:,2]-start_y)<0.01*dx)
            ind = np.intersect1d(indx[0],indy[0])

            mesh.data[ind,1] = mesh_deform_Ind(mesh.data[ind,1],0.,interface_z,1.0,1)
    
with mesh.deform_mesh():
    
    mesh.data[:,0] = mesh_deform(mesh.data[:,0],0.,0.2,0)
    mesh.data[:,2] = mesh_deform(mesh.data[:,2],0.,0.2,2)
    
    
velocityField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)
pressureField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)

previousVmMesh = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)
velAMesh  =  uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)
vel_effMesh  =  uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)

viscosityField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)

strainRateField = mesh.add_variable(nodeDofCount=1)

pressureField.data[:] = 0.
velocityField.data[:] = [0., 0., 0.]


    
# # The swarm

part_per_cell = 125
swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)

previousVm =  swarm.add_variable( dataType="double", count=3 )
velA  =  swarm.add_variable( dataType="double", count=3 )
vel_eff  =  swarm.add_variable( dataType="double", count=3 )

# Initialise the 'materialVariable' data to represent different materials.
materialV = 1   # ice
materialA = 2   # Air

if(LoadFromFile == False): 
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm=swarm, particlesPerCell=part_per_cell)
    #swarmLayout = uw.swarm.layouts.PerCellGaussLayout( swarm=swarm, gaussPointCount=5 )
    swarm.populate_using_layout(layout=swarmLayout)

# create pop control object
pop_control1 = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=part_per_cell)

# ### Create a particle advection system
# Note that we need to set up one advector systems for each particle swarm (our global swarm and a separate one if we add passive tracers).
advector1 = uw.systems.SwarmAdvector(swarm=swarm, velocityField=velocityField, order=2)



# ## Surfaceswarm in order to measure the change in surface height

# In[ ]:


## swarms to track the deformation
surfaceSwarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
previousVmB =  surfaceSwarm.add_variable( dataType="double", count=3 )
velB =  surfaceSwarm.add_variable( dataType="double", count=3 )
vel_effB  =  surfaceSwarm.add_variable( dataType="double", count=3 )

# create pop control object
# pop_control3 = uw.swarm.PopulationControl(surfaceSwarm)

# create wireframe
if(LoadFromFile == False): 
    p = (int(part_per_cell**0.33333) + 1) * resX
    surfacePoints = np.array(np.meshgrid(np.linspace(minX, maxX, p), H0, np.linspace(minZ, maxZ, p))).T.reshape(-1, 3)
    surfaceSwarm.add_particles_with_coordinates(surfacePoints)

# create advector
advector3 = uw.systems.SwarmAdvector(swarm=surfaceSwarm, velocityField=velocityField, order=2)



# Tracking different materials

materialVariable = swarm.add_variable(dataType="int", count=1)
directorVector   = swarm.add_variable( dataType="double", count=3)

if(LoadFromFile == False): 
    
    coord = fn.input()
    # conditions = [ (coord[1] > H0, materialA),
    #                (coord[1] > fn_basis, materialV),
    #                (True, materialR)]       
    conditions = [ (coord[1] > H0, materialA),
                   (True, materialV)]   

    materialVariable.data[:] = fn.branching.conditional(conditions).evaluate(swarm)


if(LoadFromFile == True):    
    surfaceSwarm.load(inputPath+"surfaceSwarm"+str(step).zfill(4))
    swarm.load(inputPath+"swarm"+str(step).zfill(4))
    materialVariable.load(inputPath+"materialVariable"+str(step).zfill(4))   

# In[ ]:


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
kWalls = mesh.specialSets["MinK_VertexSet"] + mesh.specialSets["MaxK_VertexSet"]

# Surface points
top = mesh.specialSets["MaxJ_VertexSet"]
botSet = mesh.specialSets['MinJ_VertexSet']

# Dirichlet
condition = uw.conditions.DirichletCondition(variable = velocityField, indexSetsPerDof=(botSet, botSet, botSet))

velocityField.data[:] = [0., 0., 0.]


# # Strainrate

# In[ ]:


strainRateTensor = fn.tensor.symmetric(velocityField.fn_gradient)
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateTensor)+1e-18


# # Effective viscosity, density and gravity

# In[ ]:


viscosityFnRock = fn.misc.constant(1e16 / 31556926)
viscosityFnAir = fn.misc.constant(1e11 / 3.1536e7)

minViscosityIceFn = fn.misc.constant(1e+11 / 31556926)
maxViscosityIceFn = fn.misc.constant(1e+16 / 31556926)
viscosityFnIceBase = 0.5 * A ** (-1.) #* (strainRate_2ndInvariantFn**((1.-n) / float(n)))
#viscosityFnIce = fn.misc.max(fn.misc.min(viscosityFnIceBase, maxViscosityIceFn), minViscosityIceFn)

#viscosityFnBase = cell_height * beta_square

viscosityMap = {
                materialV: viscosityFnIceBase,
                materialA: 1.*viscosityFnAir,
               }

viscosity2Map = { materialV: 0., 
                  materialA: 0.999*viscosityFnAir }

viscosityFn = fn.branching.map( fn_key=materialVariable, mapping=viscosityMap )


secondViscosityFn  = fn.branching.map( fn_key = materialVariable, 
                                       mapping = viscosity2Map )

orientation = 0. * math.pi / 180.0 
directorVector.data[:,0] = math.cos(orientation)
directorVector.data[:,1] = math.sin(orientation)
directorVector.data[:,2] = math.sin(orientation)  


densityFnIce = fn.misc.constant( ice_density )
densityFnRock = fn.misc.constant( 2700. )
densityFnAir = fn.misc.constant (0.)

densityMap = {
                materialV: densityFnIce,
                materialA: densityFnAir,
             }

densityFn = fn.branching.map( fn_key=materialVariable, mapping=densityMap )

surf_inclination = alpha
z_hat = (math.sin(surf_inclination), - math.cos(surf_inclination), 0.)

buoyancyFn = densityFn * z_hat * 9.81


# In[ ]:



# In[ ]:


devStressFn = 2.0 * viscosityFn * strainRateTensor
shearStressFn = strainRate_2ndInvariantFn * viscosityFn * 2.0

stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    voronoi_swarm=swarm,
    conditions=condition,
    fn_viscosity=viscosityFn,
    fn_bodyforce=buoyancyFn,
    _fn_viscosity2 = secondViscosityFn,
    _fn_director   = directorVector, 
)

solver = uw.systems.Solver (stokes)

#solver.set_inner_method ("lu")
#solver.set_inner_method ("superlu")
#solver.set_inner_method ("mumps")
#solver.set_inner_method ("superludist")
solver.set_inner_method ("mg")
#solver.set_inner_method ("nomg")

#solver.set_penalty (1.0e5)  # higher penalty = larger stability

#solver.options.scr.ksp_rtol = 1.0e-3
#solver.set_penalty( 1.e6 )
#nl_tol = 1.e-2 # standard value

surfaceArea = uw.utils.Integral( fn=1.0, mesh=mesh, integrationType='surface', surfaceIndexSet=top)
surfacePressureIntegral = uw.utils.Integral( fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=top)

      
def calibrate_pressure():

    global pressureField
    global surfaceArea
    global surfacePressureIntegral

    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate() 
    pressureField.data[:] -= p0 / area




# # Mainloop and flow function

# In[ ]:
# previousVm.data[:] = 0.
# previousVmB.data[:] = 0.

def update():
    

    dt = 10*advector1.get_max_dt()
    # Advect using this timestep size.
    advector1.integrate(dt) # the swarm
    # advector2.integrate(dt) # the wireframe swarm
    advector3.integrate(dt) # the surface swarm
    
#     velA.data[:] = velocityField.evaluate(swarm)  
#     vel_eff.data[:] = 1./2.*(velA.data[:]+1.*previousVm.data[:])  
    
#     previousVm.data[:] = np.copy(velA.data[:])     

#     with swarm.deform_swarm():
#         swarm.data[:] += vel_eff.data[:]*dt 
    
#     velB.data[:] = velocityField.evaluate(surfaceSwarm) 
#     vel_effB.data[:] = 1./2.*(velB.data[:]+1.*previousVmB.data[:]) 
    
#     previousVmB.data[:] = np.copy(velB.data[:]) 
#     with surfaceSwarm.deform_swarm():
#         surfaceSwarm.data[:] += vel_effB.data[:]*dt
        
#     velAMesh.data[:] = velocityField.evaluate(mesh)  
#     vel_effMesh.data[:] = 1./2.*(velAMesh.data[:]+1.*previousVmMesh.data[:])  
    
#     previousVmMesh.data[:] = np.copy(velAMesh.data[:])     
#     with mesh.deform_mesh():
#         mesh.data[:] += vel_effMesh.data[:]*dt 

    # particle population control
    pop_control1.repopulate()
    #pop_control2.repopulate()
    #pop_control3.repopulate()
    
    if rank==0:
        print('step=',step,'time=',timestep,'dt=',dt,'exec_time=',exec_time)        
    comm.Barrier()
    
    return timestep+dt, step+1


# In[ ]:


while step<nsteps:
    
    exec_time = time.time()
#     solver.solve(nonLinearIterate=False, nonLinearTolerance=1e-3, nonLinearMaxIterations=20,callback_post_solve=calibrate_pressure)
    solver.solve(nonLinearIterate=False,callback_post_solve=calibrate_pressure)
    
    exec_time = time.time() - exec_time
    
    if step %step_out == 0 or step == nsteps-1:
        
#         meshVis = uw.mesh.MeshVariable(mesh, 1)
#         projectorVis = uw.utils.MeshVariable_Projection( meshVis, viscosityFn, type=0 )
#         projectorVis.solve()
        mesh.save(outputPath+"mesh"+str(step).zfill(4)) 

        #print (f"Particle local count rank {rank} {swarm.particleLocalCount}")
    
        velocityField.save(outputPath+"velocityField"+str(step).zfill(4))
        surfaceSwarm.save(outputPath+"surfaceSwarm"+str(step).zfill(4))
#         meshVis.save(outputPath+"viscosity"+str(step).zfill(4))

        #swarm.save(inputPath+"swarm"+str(step).zfill(4))
        materialVariable.save(inputPath+"materialVariable"+str(step).zfill(4))  
    
#         meshFileHandle = mesh.save(outputPath+"Mesh.h5")
#         swarmFileHandle = surfaceSwarm.save(outputPath+"surfaceSwarm.h5")

#         filename = outputPath+"/velocityField"+str(step).zfill(4)
#         vFH      = velocityField.save(filename+".h5")
#         velocityField.xdmf(filename, vFH, "velocity", meshFileHandle, "Mesh",timestep )

#         filename = outputPath+"/meshViscosity"+str(step).zfill(4)
#         visFH      = meshVis.save(filename+".h5")
#         meshVis.xdmf(filename, visFH, "viscosity", meshFileHandle, "Mesh", timestep )

#         filename = outputPath+"/surfaceSwarm"+str(step).zfill(4)
#         sfFH      = surfaceParticle.save(filename+".h5")
#         surfaceParticle.xdmf(filename, sfFH, "surfaceParticle", swarmFileHandle, "Swarm", timestep )

        if rank==0:
            np.save(outputPath+"time"+str(step).zfill(4),timestep)
            print(f'\n -------  step: {step}  ------- \n')

        comm.Barrier()
        
    timestep, step = update()
