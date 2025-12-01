function dydt = dydt_varbal(t,y,g,alpha,tau_q)

%d/dt(h_i)=-h_i + sum_j W_{ij} tanh(gh_j)
%d/dt(q_i)=(x_i^2-q_i)/tau_q
%d/dt(w_ij)=-alpha*(x_i^2-q_i+x_j^2-q_j)(w_ij)

%This inverts a set of equations with N^2+2N entries
%M=N^2+2N = N^2 + 2N + 1 - 1 = (N+1)^2 - 1 
%(N+1)^2-(1+M)=0=(N+1+sqrt(1+M))(N+1-sqrt(1+M))
%N=sqrt(1+M)-1
N=round(sqrt(length(y)+1)-1);

W=reshape(y(1:N^2),[N N]);
h=y(N^2+1:N^2+N);
q=y(N^2+N+1:end);


delw=@(xi,xj,qi,qj,w) -alpha.*(xi.^2-qi+xj'.^2-qj').*w;

dW=delw(tanh(g*h),q,tanh(g*h),q,W);
dW(logical(eye(N)))=0;%no autapses

dh=-h + W*tanh(g*h)./sqrt(N);
dq=(tanh(g*h).^2-q)/tau_q;
dydt=[dW(:); dh; dq];
end

