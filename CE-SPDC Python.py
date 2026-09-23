import numpy as np
import matplotlib.pyplot as plt


#Sellmeier Equations for PPKTP
#y
A1y = 2.09930
A2y = 0.922683
A3y = 0.0467695
A4y = 0.0138408

#z
Az = 2.12725
Bz = 1.18431
Cz = 5.14852*10**-2
Dz = 0.6603
Ez = 100.00507
Fz = 9.68956*10**-3

temp = 60


def n1z(λ):
    return 9.9587e-6 + 9.9228e-6/λ - 8.9603e-6/λ**2 + 4.1010e-6/λ**3
def n2z(λ):
    return -1.1882e-8 + 10.459e-8/λ - 9.8136e-8/λ**2 + 3.1481e-8/λ**3
def Δnz(λ):
    return n1z(λ)*(temp - 25) + n2z(λ)*(temp - 25)**2
def n1y(λ):
    return 6.2897e-6 + 6.3061e-6/λ - 6.0629e-6/λ**2 + 2.6486e-6/λ**3
def n2y(λ):
    return -0.14445e-8 + 2.2244e-8/λ - 3.5770e-8/λ**2 + 1.3470e-8/λ**3
def Δny(λ):
    return n1y(λ)*(temp - 25) + n2y(λ)*(temp - 25)**2
def ny(λ):
    return np.sqrt(A1y + A2y/(1 - A3y/λ**2) - A4y*λ**2) + Δny(λ)
def nz(λ):
    return np.sqrt(Az + Bz/(1 - Cz/λ**2) + Dz/(1 - Ez/(λ**2)) - Fz*λ**2) + Δnz(λ)


λ_vals = np.linspace(0.5, 1.6, 500)
plt.plot(λ_vals, ny(λ_vals), color='blue',   label='ny')
plt.plot(λ_vals, nz(λ_vals), color='magenta', label='nz')
plt.xlabel('λ (µm)')
plt.ylabel('n')
plt.title('Refractive indices of PPKTP')
plt.legend()
plt.grid(True)
plt.show()

#Poling period calculations
print("\n POLING PERIOD")

λpo = 532e-9
λso = 810e-9
λio = 1/(1/λpo - 1/λso)
c = 2.9979e8
π = np.pi

print("The central wavelength for the idler is " , λio*10**9 , " nm")

ωpo = 2*π*c/λpo
ωso = 2*π*c/λso
ωio = 2*π*c/λio

kpo = nz(λpo*10**6)*ωpo/c
kso = nz(λso*10**6)*ωso/c
kio = nz(λio*10**6)*ωio/c

Λ = (-2*π/(-kpo+kso+kio))*10**6

print("The poling period is " , Λ , " microns")


#JSI no cavity
print("\n JSI NO CAVITY")

Δt = 1e-9
Δν = (.441/(Δt))/(10**9)
σν = Δν/(2*np.sqrt(2*np.log(2)))

def λs(ν):
    return c/ν
def λi(ν):
    return c/ν

L = 0.4e-2
A = 1

νso = (c/λso)/(10**9)
νio = (c/λio)/(10**9)
νpo = (c/λpo)/(10**9)

print("The central frequency for the signal is ", νso," GHz")
print("The central frequency for the idler is ", νio," GHz")

Δk0 = -nz(λpo*10**6)*2*π*νpo*10**9/c + nz(λso*10**6)*2*π*νso*10**9/c + nz(λio*10**6)*2*π*νio*10**9/c + 2*π/(Λ*10**-6)
print("Delta k0 is ", Δk0)

h = .1
dkp = (2*π*nz((c/((νpo+h)*1e9))*1e6)*(νpo+h)*1e9/c
     - 2*π*nz((c/((νpo-h)*1e9))*1e6)*(νpo-h)*1e9/c) / (2*h)
dks = (2*π*nz((c/((νso+h)*1e9))*1e6)*(νso+h)*1e9/c
     - 2*π*nz((c/((νso-h)*1e9))*1e6)*(νso-h)*1e9/c) / (2*h)
dki = (2*π*nz((c/((νio+h)*1e9))*1e6)*(νio+h)*1e9/c
     - 2*π*nz((c/((νio-h)*1e9))*1e6)*(νio-h)*1e9/c) / (2*h)

def sinc(x): return np.sin(x)/x
def Δk1(Δνs, Δνi): return (dkp - dks)*Δνs + (dkp - dki)*Δνi
def Δk(Δνs,Δνi): return Δk0 + Δk1(Δνs, Δνi)
def α(Δνs,Δνi): return np.exp(-(Δνs+Δνi)**2/(4*σν**2))
def ϕ(Δνs,Δνi): return sinc((Δk(Δνs,Δνi)*L)/2)
def JSA(Δνs,Δνi): return ϕ(Δνs,Δνi)*α(Δνs,Δνi)
def JSI(Δνs,Δνi): return np.abs(JSA(Δνs, Δνi))**2


Δpm = 2.7831/(2*π*(np.abs(dki-dks)*L))
print("The phase matching bandwidth is ",Δpm, " GHz")

δν = 7
N = 300
x = np.linspace(-δν, δν, 300)
S, I = np.meshgrid(x, x)
plt.pcolormesh(S, I, α(S, I), shading='auto', cmap='viridis')
plt.xlabel("Signal (GHz)")
plt.ylabel("Idler (GHz)")
plt.title("Pump")
plt.colorbar(label="α")
plt.show()

plt.pcolormesh(S, I, ϕ(S, I), shading='auto', cmap='viridis')
plt.xlabel("Signal (GHz)")
plt.ylabel("Idler (GHz)")
plt.title("Phase Matching")
plt.colorbar(label="ϕ")
plt.show()

plt.pcolormesh(S, I, JSI(S, I), shading='auto', cmap='viridis')
plt.xlabel("Signal (GHz)")
plt.ylabel("Idler (GHz)")
plt.title("JSI")
plt.colorbar(label="Intensity")
plt.show()


#Fabry Perot Cavity
("\n FABRY PEROT CAVITY")

R1 = .999
R2 = .98
r1 = np.sqrt(R1)
r2 = np.sqrt(R2)

αs = 127e-6*100
def αr(L,R1,R2): return αs + 1/(2*L)*np.log(1/(R1*R2))

def Fnoloss(r1,r2): return π*np.sqrt(r1*r2)/(1-r1*r2)
def F(L,R1,R2): return π*np.exp(-αr(L,R1,R2)*L/2)/(1-np.exp(-αr(L,R1,R2)*L))
print("The finesse is ",F(L,R1,R2))

def νf(L): return c*10**-9/(2*L*nz(λso*10**6))
def sw(L,R1,R2): return νf(L)/(F(L,R1,R2))
def swnoloss(L,r1,r2): return νf(L)/(Fnoloss(r1,r2))
print("The spectral width is ",sw(L,R1,R2)*10**3, " MHz")
print("The spectral width with no loss is ", swnoloss(L,r1,r2)*10**3, " MHz")

Io = 1
Imax = Io/((1-r1*r2)**2)
print("The FSR is ",νf(L), " GHz" )

def Intensity(Δν,L,R1,R2): return Imax/(1+(2*F(L,R1,R2)/π)**2*(np.sin(π*Δν/νf(L)))**2)
def qso(L): return int(np.round(νso/νf(L)))
def νr(L): return qso(L)*νf(L)
def δνr(L): return νr(L) - νso
print("The mode that is closest to the signal frequency is number ", qso(L))

δν2 = 25
ν_vals = np.linspace(-δν2, δν2, 10000)
plt.plot(ν_vals, Intensity(ν_vals,L,R1,R2), color='blue',   label='Intensity')
plt.xlabel('ν (GHz)')
plt.ylabel('Intensity')
plt.title('Intensity of a Fabry Perot Cavity')
plt.legend()
plt.grid(True)
plt.show()

L_vals = np.linspace(.02, 2, 500)
plt.plot(L_vals, νf(L_vals) , color='blue',   label='Intensity')
plt.xlabel('Crystal Length (cm)')
plt.ylabel('FSR (GHz)')
plt.title('FSR vs. Crystal length')
plt.legend()
plt.grid(True)
plt.show()

R2_vals = np.linspace(0.9,1,500)
LL, RR = np.meshgrid(L_vals, R2_vals)
plt.pcolormesh(LL, RR, np.log10(sw(LL,R1,RR)), shading='auto', cmap='viridis')
plt.xlabel("Crystal Length (cm)")
plt.ylabel("R2")
plt.title("Spectral Width")
cbar = plt.colorbar()
cbar.set_label("Spectral Width (GHz)")
ticks = cbar.get_ticks()
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{10**t:.2e}" for t in ticks])
plt.show()


#JSI With Cavity
print("\n JSI WITH CAVITY")

def JSACavity(Δνs,Δνi): return ϕ(Δνs,Δνi)* α(Δνs,Δνi)*Intensity(Δνs,L,R1,R2)
def JSICavity(Δνs,Δνi): return np.abs(JSACavity(Δνs, Δνi))**2

x = np.linspace(-δν2/10, δν2/10, 500)
S, I = np.meshgrid(x, x)
plt.pcolormesh(S, I, JSICavity(S, I), shading='auto', cmap='viridis')
plt.xlabel("Signal (GHz)")
plt.ylabel("Idler (GHz)")
plt.title("JSI with Cavity")
plt.colorbar(label="Intensity")
plt.show()

def JSACavityShifted(Δνs,Δνi): return ϕ(Δνs,Δνi)* α(Δνs,Δνi)*Intensity(Δνs-δνr(L),L,R1,R2)
def JSICavityShifted(Δνs,Δνi): return np.abs(JSACavityShifted(Δνs, Δνi))**2

x = np.linspace(-δν2, δν2, 1000)
S, I = np.meshgrid(x, x)
plt.pcolormesh(S, I, JSICavityShifted(S, I), shading='auto', cmap='viridis')
plt.xlabel("Signal (GHz)")
plt.ylabel("Idler (GHz)")
plt.title("JSI with Cavity (shifted)")
plt.colorbar(label="Intensity")
plt.show()


#ABCD Matrices for Hemispherical Cavity
print("\n ABCD MATRICES FOR HEMISPHERICAL CAVITY")

RC = 1e-2
m1 = np.array([[1,0],[0,1]])
def m2(L): return np.array([[1,L],[0,1]])
def m3(RC): return np.array([[1,0],[-2/RC,1]])
nair = 1

def ABCD(RC,L): return m1 @ m2(L) @ m3(RC) @ m2(L)
def mA(RC,L): return ABCD(RC,L)[0,0]
def mB(RC,L): return ABCD(RC,L)[0,1]
def mC(RC,L): return ABCD(RC,L)[1,0]
def mD(RC,L): return ABCD(RC,L)[1,1]

def stability(RC1,RC2,L): return (1-L/RC1)*(1-L/RC2)
def hemiStability(RC2,L): return (1-L/RC2)
print("The stability is ", hemiStability(RC,L))

def RC1(RC,L): return 2*mB(RC,L)/(-(mA(RC,L)-mD(RC,L)))
def W1(RC,L): return np.sqrt(λso/(nz(λso*10**6)*π))*np.sqrt(np.abs(mB(RC,L))/(np.sqrt(1-((mA(RC,L)+mD(RC,L))/2)**2)))
def z0(RC,L): return nz(λso*10**6)*π*W1(RC,L)**2/λso
def RC2(z,RC,L): return z*(1+((z0(RC,L))/z)**2)
def W2(z,RC,L): return W1(RC,L)*(1+(z/z0(RC,L))**2)**(1/2)
def Div1(RC,L): return λso/(π*nair*W1(RC,L))

print("The beam divergence is ",Div1(RC,L), " radians")
print("The radius of curvature of the beam at the firt mirror is ",RC1(RC,L))
print("The beam radius at the planar mirror is ",W1(RC,L)*10**6, " microns")
print("The radius of curvature of the beam at the second mirror is ",RC2(L,RC,L)*10**2," cm")
print("The beam radius at the curved mirror is ",W2(L,RC,L)*10**6," microns")

z_vals = np.linspace(0, L, 500)
plt.plot(z_vals, W2(z_vals,RC,L)*10**6, color='blue',   label='Intensity')
plt.xlabel('z (m)')
plt.ylabel('Beam Radius (microns)')
plt.title('Beam Radius vs. Z')
plt.legend()
plt.grid(True)
plt.show()

rc_vals = np.linspace(0.50001e-2, 1e-2, 500)
plt.plot(rc_vals, np.array([W1(rc, L) for rc in rc_vals]) * 1e6, color='blue',   label='Planar')
plt.plot(rc_vals, np.array([W2(L, rc, L) for rc in rc_vals]) * 1e6, color='magenta', label='Curved')
plt.xlabel('Radius of Curvature (m)')
plt.ylabel('Beam radius (microns)')
plt.title('Beam Radius for Concentric -> Confocal Cavity')
plt.legend()
plt.grid(True)
plt.show()

l_vals = np.linspace(0, 2*L, 500)
plt.plot(l_vals, np.array([W1(RC, l) for l in l_vals])*1e6, color='blue',   label='Intensity')
plt.xlabel('Cavity Length (m)')
plt.ylabel('Beam Radius (microns)')
plt.title('Beam Radius vs. Cavity Length')
plt.legend()
plt.grid(True)
plt.show()


#Beam Divergence after Cavity
print("\n BEAM DIVERGENCE AFTER CAVITY")

def m4(RC): return np.array([[1,0],[(nz(λso*10**6)-nair)/(nair*RC),nz(λso*10**6)/nair]])
def ABCDrefraction(RC): return m4(RC)
def mAr(RC): return ABCDrefraction(RC)[0,0]
def mBr(RC): return ABCDrefraction(RC)[0,1]
def mCr(RC): return ABCDrefraction(RC)[1,0]
def mDr(RC): return ABCDrefraction(RC)[1,1]

def qp1(RC,L): return L+1j*z0(RC,L)
def qp2(RC,L): return (mAr(RC)*qp1(RC,L)+mBr(RC))/(mCr(RC)*qp1(RC,L)+mDr(RC))
def zp2(RC,L): return -np.real(qp2(RC,L))
def z02(RC,L): return np.imag(qp2(RC,L))
def W12(RC,L): return np.sqrt(λso*z02(RC,L)/(π*nair))
def W22(z,RC,L): return W12(RC,L)*(1+(z/z02(RC,L))**2)**(1/2)
def Div2(RC,L): return λso/(π*nair*W12(RC,L))

print("The beam divergence after exiting the cavity is ",Div2(RC,L)," radians")

z_vals = np.linspace(-L, 6*L, 500)
plt.plot(z_vals, W2(z_vals,RC,L)*10**6, color='blue',   label='Inside Cavity')
plt.plot(z_vals, W22(z_vals-L-zp2(RC,L),RC,L)*10**6, color='magenta', label='Outside Cavity')
plt.xlabel('z(m)')
plt.ylabel('Beam Radius (microns)')
plt.title('Beam Radius Through Interface')
plt.legend()
plt.grid(True)
plt.axvline(L, color='red', linestyle='--')
plt.show()

L_vals = np.linspace(L/4, 2*L, 500)
plt.plot(L_vals, np.array([Div1(RC,l) for l in L_vals]) , color='blue',   label='Inside Cavity')
plt.plot(L_vals, np.array([Div2(RC,l) for l in L_vals]), color='magenta', label='Outside Cavity')
plt.xlabel('Crystal Length')
plt.ylabel('Beam Divergence (rad)')
plt.title('Divergence vs. Crystal Length')
plt.legend()
plt.grid(True)
plt.show()

RC_vals = np.linspace(1e-2,2e-2,100)
L_vals = np.linspace(0.1e-2,1e-2,100)
LL, RR = np.meshgrid(L_vals, RC_vals)
plt.pcolormesh(LL, RR, np.log10(np.array([[Div2(rc, l) for l in L_vals] for rc in RC_vals])), shading='auto', cmap='viridis')
plt.xlabel("Crystal Length (m)")
plt.ylabel("Radius of Curvature")
plt.title("Beam Divergence")
cbar = plt.colorbar()
cbar.set_label("Beam Divergence (rad)")
ticks = cbar.get_ticks()
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{10**t:.2e}" for t in ticks])
plt.show()

#Modes
print("\n MODES")

def Δζ(RC,L): return np.arctan(L/z0(RC,L))
def Δζ2(RC,L): return np.arccos(np.sqrt(1-L/RC))
def νHG(q,z,m,RC,L): return q*νf(L)+(z+m+1)*(Δζ(RC,L)/π)*νf(L)
def shift(q1,z1,m1,q2,z2,m2,RC,L): return νHG(q2,z2,m2,RC,L)-νHG(q1,z1,m1,RC,L)

print("The frequency shift from TEM00 to TEM01 is ", shift(1,0,0,1,0,1,RC,L)," GHz")
print("TEM00 resonance: ",νHG(0,0,0,RC,L)," GHz")
print("TEM10 resonance: ",νHG(0,1,0,RC,L)," GHz")
print("TEM20 resonance: ",νHG(0,2,0,RC,L)," GHz")


ν_vals = np.linspace(-20, 20, 5000)
plt.plot(ν_vals, Intensity(ν_vals-νHG(0,0,0,RC,L),L,R1,R2), color='red',   label='TEM00')
plt.plot(ν_vals, Intensity(ν_vals-νHG(0,1,0,RC,L),L,R1,R2)*0.85, color='orange',   label='TEM10')
plt.plot(ν_vals, Intensity(ν_vals-νHG(0,2,0,RC,L),L,R1,R2)*0.7, color='yellow',   label='TEM20')
plt.plot(ν_vals, Intensity(ν_vals-νHG(0,3,0,RC,L),L,R1,R2)*0.55, color='green',   label='TEM30')
plt.plot(ν_vals, Intensity(ν_vals-νHG(0,4,0,RC,L),L,R1,R2)*0.4, color='blue',   label='TEM40')
plt.plot(ν_vals, Intensity(ν_vals-νHG(0,5,0,RC,L),L,R1,R2)*0.25, color='purple',   label='TEM50')
plt.xlabel('ν (GHz)')
plt.ylabel('Intensity')
plt.title('Intensity of Higher Order Spatial Modes')
plt.legend()
plt.grid(True)
plt.show()

def TotalIntensity(ν,RC,L,R1,R2): return Intensity(ν-νHG(0,0,0,RC,L),L,R1,R2)+Intensity(ν-νHG(0,1,0,RC,L),L,R1,R2)*.5+Intensity(ν-νHG(0,2,0,RC,L),L,R1,R2)*.5+Intensity(ν-νHG(0,3,0,RC,L),L,R1,R2)*.5+Intensity(ν-νHG(0,4,0,RC,L),L,R1,R2)*.5+Intensity(ν-νHG(0,5,0,RC,L),L,R1,R2)*.5

#JSI with Higher Order Modes
print("JSI WITH HIGHER ORDER MODES")

def JSAHigherModes(Δνs,Δνi): return ϕ(Δνs,Δνi)*α(Δνs,Δνi)*TotalIntensity(Δνs,RC,L,R1,R2)
def JSIHigherModes(Δνs,Δνi): return np.abs(JSAHigherModes(Δνs,Δνi))**2

x = np.linspace(-δν2/3, δν2/3, 1000)
S, I = np.meshgrid(x, x)
plt.pcolormesh(S, I, JSIHigherModes(S, I), shading='auto', cmap='viridis')
plt.xlabel("Signal (GHz)")
plt.ylabel("Idler (GHz)")
plt.title("JSI with Higher Order Spatial Modes")
plt.colorbar(label="Intensity")
plt.show()

#Stability
print("STABILITY")

def Δnztemp(λ,T): return n1z(λ)*(T-25)+n2z(λ)*(T-25)**2
def nztemp(λ,T): return np.sqrt(Az+Bz/(1-Cz/(λ**2)))+Δnztemp(λ,T)

aαz = 0.02e-6
bαz = 6.5e-9

def Lz(T): return 4e-3*(1+aαz*(T-31.85)+(1/2)*bαz*(T-31.85)**2)

T_vals = np.linspace(20, 50, 500)
plt.plot(T_vals, nztemp(.81,T_vals), color='blue',   label='Temperature')
plt.xlabel('Temperature (C)')
plt.ylabel('Index of Refraction')
plt.title('Index of Refraction vs. Temperature')
plt.legend()
plt.grid(True)
plt.show()

T_vals = np.linspace(30, 60, 500)
plt.plot(T_vals, Lz(T_vals), color='blue',   label='Temperature')
plt.xlabel('Temperature (C)')
plt.ylabel('Length (m)')
plt.title('Length vs. Temperature')
plt.legend()
plt.grid(True)
plt.show()





# α λ π ν δ Δ σ ω ϕ ξ ζ