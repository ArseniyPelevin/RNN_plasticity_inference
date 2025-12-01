%Trying to maintain chaotic network dynamics
%d/dt(h_i)=-h_i + sum_j W_{ij} tanh(gh_j)
%d/dt(q_i)=(x_i^2-q_i)/tau_q
%d/dt(w_ij)=-alpha*(x_i^2-q_i+x_j^2-q_j)(w_ij)
N=300;
g=2;
alpha=.5;
tau_q=30;
Tm=800;

h0=normrnd(0,1,[N 1]);
%W0=rand([N N]);
W0=normrnd(0,1,[N N]);
W0(W0<-1)=-1;
W0(W0>1)=1;
W0(logical(eye(N)))=0; %no autapses
q0=.5*ones(size(h0));

y0=[W0(:); h0; q0];
[t,y]=ode45(@(t,y) dydt_varbal(t,y,g,alpha,tau_q),[0 Tm],y0);

%Measure Lyapunov exponent
%pick size of delta vector
sigh=1e-6;
sigW=0;
delta_h=normrnd(0,sigh^2,size(h0));
delta_W=normrnd(0,sigW^2,size(W0));
delta_q=0;
%delta initial conditions
hd=h0+delta_h;
Wd=W0+delta_W;
qd=q0;
%delta simulation
yd=[Wd(:); hd; qd];
[td,yd]=ode45(@(t,y) dydt_varbal(t,y,g,alpha,tau_q),[0 Tm],yd);

%Compare at the same times
y_interp=@(tc) interp1(t,y,tc);
yd_interp=@(tc) interp1(td,yd,tc);

t_compare=(0:.1:Tm);
Delta=vecnorm(y_interp(t_compare)-yd_interp(t_compare),2,2);
cutoff=1e6;
tvals=Delta<Delta(1).*cutoff;
fit_param=polyfit(t_compare(tvals),log(Delta(tvals)),1);

h=y(:,N^2+1:N^2+N);
figure;
imagesc(t,1:N,tanh(g*h)');
xlabel('time')
ylabel('N');
title('h')
clim([-1 1]);
colormap redblue;
colorbar;

q=y(:,N^2+N+1:N^2+2*N);
figure;
imagesc(t,1:N,q');
xlabel('time')
ylabel('N');
title('q')
clim([-1 1]);
colormap redblue;
colorbar;


W=y(:,1:min(1000,N^2));
figure;
imagesc(t,1:min(1000,N^2),W');
xlabel('time')
ylabel('N');
title('W');
colormap redblue;
clim([-1 1]);
colorbar;

figure;
hold on;
semilogy(t_compare,Delta);
semilogy(t_compare(tvals),exp(fit_param(1).*t_compare(tvals)+fit_param(2)),'r--');
ylabel('Delta');
xlabel('time');
set(gca,'tickdir','out');
set(gca,'yscale','log');
box off;
legend('simulation',['\lambda=' num2str(fit_param(1))]);