# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# This model of Exp E is from Feb 2022. 
#
# This is the 'free slip' version, as described by Pattyn 2008. 
#
# **Important note:** there is still no-slip for most of the model area! Only the section where 2200 <= x <= 2500 is free-slip!
#
# Due to the irregular and mostly oblque basal topography  we can not simply use the common free-slip condition - which would have to enable slip in x _and_ y. This can not work, obviously.
#
# Idea: have a basal layer with a viscosity of 0.
#
# It models the glacier described in Pattyn (...) as a deformed mesh. Some ingenuity is necessary for this to work, so maybe there will be another version necessary. Obe that just uses particles.

# # Basic python imports and model settings

# +
from underworld import function as fn
import underworld as uw

import underworld.visualisation as vis

import matplotlib.pyplot as pyplot
import numpy as np
from scipy.spatial import distance

import math
import os
import sys

import time

from scipy.signal import savgol_filter

g = 9.81
ice_density = 910.

A = 1e-16
n = 3.
# -

# ## Get topo info from file

# +
topo = np.genfromtxt("topo.csv")
topo_base = topo[:, 0:-2]
topo_surf = topo[:, 0:-1]
topo_surf = np.delete(topo_surf, obj=1, axis=1)

pyplot.plot(topo_base[:,0], topo_base[:,-1])
pyplot.plot(topo_surf[:,0], topo_surf[:,-1])

pyplot.show()

# +
resX = topo_base.shape[0] - 1 
resY = topo_base.shape[0] - 1 + 1 # the trailing +1 because we need a higher resolution compared with the no-slip experiments

air_height = 50.

DeltaY = np.max(topo_surf[:,1]) + air_height - np.min(topo_base[:,1]) 

maxX = np.max(topo_surf[:,0])
maxY = np.max(topo_surf[:,1]) + air_height

minX = np.min(topo_base[:,0])
minY = np.min(topo_base[:,1]) - DeltaY / (resY - 1)

print(f"resX: {resX}, resY: {resY}" )
print(f"minX: {minX}, minY: {minY}" )
print(f"maxX: {maxX}, maxY: {maxY}" )

cell_height = maxY / resY
cell_width = maxX / resX
# -

# # The mesh

# +
elementType = "Q1/dQ0"

mesh = uw.mesh.FeMesh_Cartesian(elementType=(elementType),
                                elementRes=(resX, resY),
                                minCoord=(minX, minY),
                                maxCoord=(maxX, maxY),
                                periodic=[False, False])

submesh = mesh.subMesh

velocityField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=mesh.dim)
pressureField = uw.mesh.MeshVariable(mesh=mesh, nodeDofCount=1)

pressureField.data[:] = 0.
velocityField.data[:] = [0., 0.]
# -

# ## Deform the mesh

figMesh = vis.Figure(figsize=(1200,600))
figMesh.append( vis.objects.Mesh(mesh, nodeNumbers=False))
figMesh.show()

# +
dx = (maxX - minX) / resX
dy = (maxY - minY) / resY

def mesh_deform_base_Ind(section, fixPoint_index, fixPoint, mi):
    
    section[fixPoint_index] = fixPoint
    seqN = len(section)
    
    for index in range(len(section)):
        
        maxCoord = np.max(section)
        minCoord = np.min(section)
            
        section[index] = fixPoint + (index-fixPoint_index)*(maxCoord-fixPoint) / (seqN-fixPoint_index-1)
        zz_pow = (section[index] - fixPoint)**mi
        zz_pow_max = (maxCoord - fixPoint)**mi
        section[index] =fixPoint + (section[index]-fixPoint) * zz_pow / zz_pow_max    
            
    return (section)

with mesh.deform_mesh():
    
    for indexx in range(resX + 1):

            start_x = dx * indexx

            interface_y = topo_base[indexx, 1]

            ind = np.where( abs(mesh.data[:, 0] - start_x) < 0.01*dx )
            mesh.data[ind[0][1:],1] = mesh_deform_base_Ind(mesh.data[ind[0][1:], 1], 0, interface_y, 0.2)
            mesh.data[ind[0][0],1] = interface_y - dy
            
# -

figMesh.show()

# # A material swarm

# +
# Initialise the 'materialVariable' data to represent different materials.
materialV = 0  	# ice, isotropic
materialA = 1   # air
materialS = 2   # soft ('0'-viscosity) material
materialH = 3   # hard ('inf'-viscosity) material

part_per_cell = 50

swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout(swarm=swarm, particlesPerCell=part_per_cell)
swarm.populate_using_layout(layout=swarmLayout)

materialVariable = swarm.add_variable(dataType="int", count=1)

coord = fn.input()

materialVariable.data[:] = materialA

iceBody = fn.shape.Polygon( np.append(topo_surf, topo_base[::-1], axis=0) )
slipperyBody = fn.shape.Polygon( np.append(topo_base, topo_base[::-1] - (0, 50), axis=0) )

for index in range( len(swarm.particleCoordinates.data) ):

    co = swarm.particleCoordinates.data[index][:]

    if iceBody.evaluate (tuple (co)):
        materialVariable.data[index] = materialV
    elif slipperyBody.evaluate(tuple (co)):
        if 2200 <= co[0] <= 2500:
            materialVariable.data[index] = materialS
        else:
            materialVariable.data[index] = materialH

measurementSwarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)

# create pop control object
pop_control1 = uw.swarm.PopulationControl(swarm, aggressive=True, particlesPerCell=part_per_cell)
advector1 = uw.systems.SwarmAdvector(swarm=swarm, velocityField=velocityField, order=2)

pop_control2 = uw.swarm.PopulationControl(measurementSwarm)
advector2 = uw.systems.SwarmAdvector(swarm=measurementSwarm, velocityField=velocityField, order=2)
# -

figMaterials = vis.Figure(figsize=(1200,600))
figMaterials.append(vis.objects.Points(swarm, materialVariable, pointSize=1.0, rulers=True))
figMaterials.show()

# # Boundary conditions

# +
sideWalls = mesh.specialSets["MaxJ_VertexSet"] + mesh.specialSets["MinJ_VertexSet"]
vertWalls = mesh.specialSets["MaxI_VertexSet"] + mesh.specialSets["MinI_VertexSet"]

botWall = mesh.specialSets["MinJ_VertexSet"]
surfWalls = sideWalls = mesh.specialSets["MaxJ_VertexSet"]

# Dirichlet
condition = uw.conditions.DirichletCondition(variable = velocityField, indexSetsPerDof=(botWall, botWall))

velocityField.data[:] = [0., 0.]
# -

strainRateTensor = fn.tensor.symmetric(velocityField.fn_gradient)
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateTensor)

# # Viscosity, density, buoyancy

# +
minViscosityIceFn = fn.misc.constant(1e+6 / 3.1536e7)
maxViscosityIceFn = fn.misc.constant(1e+15 / 3.1536e7)

viscosityFnAir = fn.misc.constant(1e9 / 3.1536e7)

# viscosity of the hard (NoSlip) basal layer
viscosityFnNS = fn.misc.constant(1e23 / 3.1536e7)

# viscosity of the soft basal layer
viscosityFnS = fn.misc.constant(8e8 / 3.1536e7)

viscosityFnIceBase = 0.5 * A ** (-1./n) * (strainRate_2ndInvariantFn**((1.-n) / float(n)))
viscosityFnIce = fn.misc.max(fn.misc.min(viscosityFnIceBase, maxViscosityIceFn), minViscosityIceFn)

viscosityMap = {
                materialV: viscosityFnIce,
                materialA: viscosityFnAir,
                materialH: viscosityFnNS,
                materialS: viscosityFnS,
               }

viscosityFn = fn.branching.map( fn_key=materialVariable, mapping=viscosityMap )

devStressFn = 2.0 * viscosityFn * strainRateTensor
shearStressFn = strainRate_2ndInvariantFn * viscosityFn * 2.0

densityFnAir = fn.misc.constant( 0.001 )
densityFnIce = fn.misc.constant( ice_density )

densityMap = {
                materialA: densityFnAir,
                materialV: densityFnIce,
                materialH: densityFnIce,
                materialS: densityFnIce,
             }

densityFn = fn.branching.map(fn_key=materialVariable, mapping=densityMap)

buoyancyFn = densityFn * (0., -1.) * 9.81
# -

devStressFn = 2.0 * viscosityFn * strainRateTensor
shearStressFn = strainRate_2ndInvariantFn * viscosityFn * 2.0

# # Solver

# +
stokes = uw.systems.Stokes(
    velocityField=velocityField,
    pressureField=pressureField,
    voronoi_swarm=swarm,
    conditions=condition,
    fn_viscosity=viscosityFn,
    fn_bodyforce=buoyancyFn,
)

solver = uw.systems.Solver(stokes)

solver.set_inner_method("mumps")

surfaceArea = uw.utils.Integral( fn=1.0, mesh=mesh, integrationType='surface', surfaceIndexSet=surfWalls)
surfacePressureIntegral = uw.utils.Integral( fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=surfWalls)

def calibrate_pressure():

    global pressureField
    global surfaceArea
    global surfacePressureIntegral

    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate() 
    pressureField.data[:] -= p0 / area

    print (f'Calibration pressure {p0 / area}')

# test it out
try:
    exec_time = time.time()
    solver.solve(nonLinearIterate=True, callback_post_solve=calibrate_pressure)
    exec_time = time.time() - exec_time

    # print full stats to a file
    solver.print_stats()
except:
    print("Solver died early..")
    exit(0)
    
print (f'Solving took: {exec_time} seconds')
# -

figVel = vis.Figure(figsize=(1200,600))
figVel.append(vis.objects.Surface(mesh, fn.math.sqrt(fn.math.dot(velocityField, velocityField)), colourBar=True ))#fn_mask = materialVariable))
figVel.show()

# # Output

# +
#### Filename
outputFile = os.path.join(os.path.abspath("."), "jla"+"1e001"+ ".csv")
print(outputFile)

#### Smooth the stress
meshStressTensor = uw.mesh.MeshVariable(mesh, 3)
projectorStress = uw.utils.MeshVariable_Projection( meshStressTensor, devStressFn, type=0 )
projectorStress.solve()

#### Smooth the velocity
meshVelocity = uw.mesh.MeshVariable(mesh, 2)
projectorV = uw.utils.MeshVariable_Projection( meshVelocity, velocityField, type=0 )
projectorV.solve()

#### Smooth the pressure
meshP = uw.mesh.MeshVariable(mesh, 1)
projectorP = uw.utils.MeshVariable_Projection( meshP, pressureField, type=0 )
projectorP.solve()

#### Points
sub = 0. #cell_height
add = cell_height

surf_pos = topo_surf
base_pos = topo_base

#### Get the surface velocity
vxs = meshVelocity.evaluate(surf_pos).transpose()[0]
vys = meshVelocity.evaluate(surf_pos).transpose()[1]

vtots = np.sqrt( vxs*vxs + vys*vys )

#### Get the basal velocity
vxb = meshVelocity.evaluate(base_pos).transpose()[0]
vyb = meshVelocity.evaluate(base_pos).transpose()[1]

vtotb = np.sqrt( vxb*vxb + vyb*vyb )

#### Get the pressure
P = meshP.evaluate(base_pos).squeeze()
#P = pressureField.evaluate(base_pos).squeeze()

#### Get the shearstress
sxy = meshStressTensor.evaluate(base_pos).squeeze()[:,2]
#sxy = devStressFn.evaluate(base_pos).squeeze()[:,2]

#### plot pressure from grid / theoretical / difference
print("DeltaP")
#pyplot.plot((maxY - base_ypos[:]) * 9.81 * 910, color='red')
smoothed_2dg = savgol_filter(P, window_length = 3, polyorder = 2)
#pyplot.plot(P[ind], color='blue')
#pyplot.plot(P - (maxY - base_pos[ind, 1].squeeze()) * 9.81 * 910., color='green')
pyplot.plot(smoothed_2dg, color='black')
pyplot.show()

#### plot vx at surface
print("vxs")
smoothed_2dg = savgol_filter(vtots, window_length = 3, polyorder = 2)
#pyplot.plot(vxs[ind], color='red')
pyplot.plot(smoothed_2dg, color='black')
pyplot.show()

### plot shear stress
print("Shear stress")
smoothed_2dg = savgol_filter(sxy / 1000, window_length = 3, polyorder = 2)
#pyplot.plot (sxz[ind] / 1000., color='red')
pyplot.plot (smoothed_2dg, color='black')
pyplot.show ()

#### output to file
with open(outputFile, "w") as text_file:
    
    for i in range(0, resX+1):
        
        # Ausgabe [x] [y]
        textline = str("{:.7f}".format(surf_pos[i, 0] / maxX)) + "\t"
        
        #Ausgabe Geschwindigkeiten Surface[vx] [vy] [vz]
        textline += str("{:.7f}".format(vtots[i])) + "\t" + str("{:.7f}".format(vys[i])) + "\t"
        
        #Ausgabe Geschwindigkeiten Basis [vx] [vy]
        #textline += str("{:.7f}".format(vxb[i])) + "\t"
        
        # Scherspannung Basis Tensoren [Txz] [Tyz]
        textline += str("{:.7f}".format(sxy[i] / 1000.)) + "\t"
        
        # Ausgabe delta p
        textline += str("{:.7f}".format( (P[i] - float((surf_pos[i, 1] - base_pos[i, 1]) * 9.81 * 910 )) / 1000)) + "\n"

        text_file.write(textline)
# -


