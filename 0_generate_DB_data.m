% Generate data using Dittus-Boelter equation for heating
% Compatible with MATLAB

clear all
clc

i=1;

for Re=5400:1000:54000
    for Pr= 0.6:1:160
        Rey(i)=Re;
        Pra(i)=Pr;
        Nu(i)=0.023*(Re^0.8)*(Pr^0.4);
        i=i+1;
    end
end

A=[Rey; Pra; Nu]';

csvwrite('DB_Corr',A);