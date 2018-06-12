function [hidlw outlw terr] = backprop(tset, tslb, inihidlw, inioutlw, lr)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate

% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN

% 1. Set output matrices to initial values
	hidlw = inihidlw;
	outlw = inioutlw;
	
% 2. Set total error to 0
	terr = 0;
	
% foreach sample in the training set
	for i=1:rows(tset)
		% 3. Set desired output of the ANN
		dout = -1 * ones(1, columns(outlw));
    dout(tslb(i)) = 1;
    [tslb(i) dout];
		% Note: I prefer bipolar outputs so "negative" answer of a neuron is -1
		
		% 4. Propagate input forward through the ANN
		% remember to extend input [tset(i, :) 1]
	  hlact = [tset(i, :) 1] * hidlw;
	  hlout = actf(hlact);    
    
    olact = [hlout 1] * outlw;
	  olout = actf(olact);
  	[~, lab] = max(olout, [], 2);
    [tslb(i), lab];

		% 5. Adjust total error (just to know this value)
    terr += sumsq(dout - olout);

		% 6. Compute delta error of the output layer
		% how many delta errors should be computed here? - 2 neurons, 2 errors
    delta_error_out = 0.5 * (dout - olout) .* actdf(olout);
		
		% 7. Compute delta error of the hidden layer
		% how many delta errors should be computed here? - 5 neurons, 5 errors
    delta_error_hid = (delta_error_out * outlw(1:end-1,:)') .* actdf(hlout);
		
		% 8. Update output layer weights
    outlw += lr * [hlout 1]' * delta_error_out;
		
		% 9. Update hidden layer weights
    hidlw += lr * [tset(i, :) 1]' * delta_error_hid;
    terr;
    
	end
end