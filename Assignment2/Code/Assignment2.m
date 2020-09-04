N=20;
n_max=50;
learning_rate = 1/N;
max_datasets=30;
errors=zeros(1, max_datasets);
stabilities=zeros(1, max_datasets);
index=0;
mean_errors=zeros(1, 12);
mean_stabs=zeros(1, 12);

for a=0.25:0.25:3
    P=a*N;
    index=index+1;
    for dataset=1:max_datasets                  %Generate datasets for each value of P
        t_max=n_max*P;
        D = randn(N,P);
        weightsSt = randn(1,N);
        weightsT = randn(1,N);
        S = sign(weightsT * D);
        store=zeros(1,t_max);
        for t=1:t_max
            stab = (weightsSt * D .* S) / norm(weightsSt);                               % Calculate stabilities
            [stab_min, idx] = min(stab);                                                 % Find minimum stability
            vector = D(:,idx);                                                           % Find vector and label belonging to idx
            label = S(idx);
            old_weightsSt = weightsSt;                                                   % Store old weights for similarity checking
            weightsSt = weightsSt + learning_rate * vector' * label;                     % Modify the weights using Hebbian update step
            error = 1/pi*acos((weightsSt*weightsT')/(norm(weightsSt)*norm(weightsT)));   % Calculate generalization error
            similarity = pdist([weightsSt;old_weightsSt], 'cosine');                     % Stop if the weights do not change anymore
            if similarity < 0.001
               break;
            end
        end
        
        errors(dataset)=error;                       %store the error at the end of the training for each dataset
        stabilities(dataset)=min(stab);
    end
    
    mean_errors(index)=mean(errors);                 %store the average dataset error for each value of P
    mean_stabs(index)=mean(stabilities);             %store the average stability for each value of P
    errors=0;                                        %reset errors array for the next value of P
    stabilities=0;                                   %reset stabilities array for the next value of P
end

%plot the results

a=0.25:0.25:3;

figure;
plot(a, mean_errors)
title('Average generalization error');
xlabel('Alpha a=P/N');
ylabel('Error e');

figure;
plot(a, mean_stabs)
title('Average stability for each alpha');
xlabel('Alpha a=P/N');
ylabel('Stability k');
