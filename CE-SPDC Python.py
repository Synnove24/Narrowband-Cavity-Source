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
ν_vals = np.linspace(-δν2, δν2, 500)
plt.plot(ν_vals, Intensity(ν_vals,L,R1,R2), color='blue',   label='Intensity')
plt.xlabel('ν (GHz)')
plt.ylabel('Intensity')
plt.title('Intensity of a Fabry Perot Cavity')
plt.legend()
plt.grid(True)
plt.show()

L_vals = np.linspace(.02e-2, 2e-2, 500)
plt.plot(L_vals, νf(L_vals) , color='blue',   label='Intensity')
plt.xlabel('Crystal Length (m)')
plt.ylabel('FSR (GHz)')
plt.title('FSR vs. Crystal length')
plt.legend()
plt.grid(True)
plt.show()



# α λ π ν δ Δ σ ω