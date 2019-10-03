function SJplot(matrix,tol,option)
%   SJplot(matrix,tol,option) creates a plot of the calculated singular 
%   value spectrum or partial singular value spectrum for a matrix from 
%   the SJsingular database.
%
%   Input:
%       matrix -- either:
%          a problem structure downloaded from the SJsingular database
%          a number: 1 to the # of matrices in the collection
%          a string: the name of a matrix in the collection
%          In the last two cases SJget must be installed first from
%          http://www.math.sjsu.edu/singular/matrices/SJget.html
%       tol (optional) -- draw a horizontal line at tol
%          (the default value is  max(size(A))*eps(norm of A))
%       option (optional) -- if option = 1 create high resolution image,
%           if option = 2 create a low resolution image, suitable for
%           thumbnail picture.  The default is option = 1.
%   Note: if blue error bars appear in the plot then a singular 
%        value of A is guaranteed to be in the interval pictured by the 
%        blue bars around each of the calculated singular values.
%   Examples:
%         %if matrix information has already been downloaded
%         SJplot(Problem), shg
%         % or    (assuming that SJget is installed)
%         SJplot('Meszaros/model6'), shg
%         % or
%         tol = 1.e-5;
%         SJplot(403, tol), shg
%
%   See also SJrank.

% Written by Nikolay Botev and Leslie Foster, 9/5/2008, Version 1.0
% Copyright 2008, Leslie Foster


   % check that the input is ok
   if isstruct(matrix) 
       if isfield(matrix,'svals')
          svals = matrix.svals;
          if isfield(matrix,'svals_err')
             svals_err = matrix.svals_err;
          end
       else
           error(['Invalid matrix. The structure ', ...
               inputname(1),' is missing a svals field.'])
       end
   else
       if isempty(which('SJget'))
           error(['SJget is not installed. It is available at ',...
               'http://www.math.sjsu.edu/singular/matrices/SJget.html .'])
       end
       matrix = SJget(matrix);
       svals = matrix.svals;
       if isfield(matrix,'svals_err')
          svals_err = matrix.svals_err;
       end
   end 
   [m,n] = size(matrix.A);  
   default_tol = 0;
   if nargin < 2 || isempty(tol)
      tol = max(m,n)*eps(svals(1));
      default_tol = 1;
   end
   if nargin >= 2 && tol < 0
      error('tolerance is negative')
   end
   if nargin <= 2 || isempty(option)
       option = 1;
   end
   if option ~= 1 && option ~= 2
       error('option is not 1 or 2')
   end
   
   fullname = matrix.name;
   if ~isempty(find(svals < 0, 1))
       full_svd = 0;
   else
       full_svd = 1;
   end

   %------------------------------------------------------------
    % Nikolay and LF: generate svd plots (thumbnail or full size)
    %------------------------------------------------------------

    clf;        % LF -- to reset graph settings to default
    if full_svd % full svd plots
    
        % Create plots for full singular value spectrum
        % Using:  
        %   svals: a (column) vector with partial information about 
        %          singular values where any entry with a -1 indicates a
        %          sing. value that has not been calculated
        %   svals_err: a (column) vector with partial information about the  
        %         accuracy of the calculated singular values where any
        %         entry with a -1 indicates a sing. value that has not been
        %         calculated. Currently svals_err is not used but code to 
        %         use svals_err is included but not accessed
        %  result: plots of singular value spectrum

        warning('off','MATLAB:Axes:NegativeDataInLogAxis')
        if option == 2
            % create svd image as thumbnail
            semilogy(svals,'--rs','LineWidth',10,'MarkerEdgeColor','g', ...
                'MarkerFaceColor','g','MarkerSize',15);
            line([0 length(svals)], [1 1]*tol, 'Color', [0 0 .5]); 
                                     % marker for tolerance
            set(gca,'XTick',[],'YTick',[]);
            xlim([0 length(svals)]);
        else
            % prepare full-res svd image
            %semilogy(svals,'--rs','LineWidth',3,'MarkerEdgeColor','g', ...
            %    'MarkerFaceColor','b','MarkerSize',5);
            semilogy(1:length(svals),svals,'--rs','LineWidth',3, ...
                'MarkerEdgeColor','g', ...
                'MarkerFaceColor','g','MarkerSize',5)
            xlim([0 length(svals)]);        
            colorm = [0 .5 0];        
            line(xlim, [1 1]*tol, 'Color', colorm,'Linewidth',1); 
                               % marker for rank with MATLAB tolerance

            if ( 1 == 1)
               % to make the resolution sharper create dots at each svd
               % location
               hold on
               semilogy(1:length(svals),svals,'.','MarkerSize',1);
               hold off
            end

            % fix missing marker in Okunbor/aft01 plot
            axisv1 = axis;
            axisv1(4) = max( max(svals), axisv1(4) );
            axis(axisv1);

            set(gca, 'FontSize', 17);
 
            % Note in some cases (eg Pajek/SmaGri)the xlabel results
            %   are written to the pdf file or the matlab plot (unless the 
            %   plot is full screen).  This may occur when the y scale goes
            %   to the underflow limit.  Is this a Matlab bug?
            nsvals_zero = sum( svals == 0 ) ;
            if ( nsvals_zero == 1 )
               xlabel(['                       singular value number ',...
                      '\newline', ...
                     ' Note: one calculated singular value is 0 and ',...
                     'is not shown'],'FontSize', 14);         
            elseif (nsvals_zero > 1 )
               xlabel(['                          singular value ',...
                       'number \newline ',...
                       ' Note: ', int2str(nsvals_zero), ...
                       ' calculated singular values are 0 ', ...
                       'and are not shown'],'FontSize', 14);         
            else
                    xlabel('singular value number', 'FontSize', 14);
            end   

            % finish full-res svd image
            grid on
            line(xlim, [1 1]*tol, 'Color', colorm,'Linewidth',1); 
                               % marker for rank with MATLAB tolerance
            if default_tol
               text(1, tol, ' max(size(A)) * eps(max(svals))', 'Color',...
                    colorm,'FontSize', 14, 'VerticalAlignment', 'bottom');
            else
               text(1, tol,['tol = ',num2str(tol,'%0.5g')], 'Color',...
                    colorm,'FontSize', 14, 'VerticalAlignment', 'bottom');
            end
            title(['Singular Value Spectrum for ' fullname],...
                  'FontSize',14, 'FontWeight', 'b', 'Interpreter', 'none');
            axisv = axis;

            if ( exist('svals_err','var') && 1 == 0)
               % skip this -  the error bounds make the graph very 
               %    cluttered near smaller singular values and the errors
               %    are small relative to the default tol
               % include errbounds in plot 
               isvp=find(svals >= 0 );
               isvp = isvp(:)';
               bound_lw = 1;   % width of line in error bounds
               for it = isvp
                   if ( svals_err(it) > 0 )
                     lowbound = max( axisv(3),svals(it)-svals_err(it));
                     upbound = min(axisv(4),svals(it)+svals_err(it));
                     line([it,it],[lowbound,upbound],'Color','b',...
                         'Linewidth',bound_lw);
                     boundlength = .15;
                     if ( lowbound ~= axisv(3) )
                        line([it - boundlength,it + boundlength],...
                             [lowbound,lowbound],...
                             'Color','b','Linewidth',bound_lw);
                     end 
                     if (upbound ~= axisv(4) )
                        line([it - boundlength, it+boundlength],...
                            [upbound, upbound],...
                            'Color','b','Linewidth',bound_lw);
                     end
                   end
               end
            end   
            ylabel('calculated singular value', 'FontSize', 14);
        end
        warning('on','MATLAB:Axes:NegativeDataInLogAxis')
      
    else % partial svd plots
        
        % Create: plot of partial singular value spectrum (lvf + nb)
        % Using:  
        %   svals: a (column) vector with partial information about 
        %          singular values where any entry with a -1 indicates
        %          a sing. value that has not been calculated
        %   svals_err: a (column) vector with partial information about the  
        %         accuracy of the calculated singular values where any
        %         entry with a -1 indicates a sing. value that has not been
        %         calculated
        %   k_m1: the number of -1's to place in each string of 
        %         uncalculated sing. values (set to 3 below)
        % Produce: 
        %   svalp: a vector with each string of -1's in svals replaced by
        %         a string of k_m1 -1's
        %   svalp_err: a vector with each string of -1's in svals_err
        %         replaced by a string of k_m1 -1's
        %   isvalp: a vector whose length is the number of strings of 
        %         consecutive calculated singular values in svalp 
        %         (and svals).  isvalp(i) contains the starting index in
        %         svalp of the ith string of calculated singular values in
        %         svalp
        %   nsvalp: a vector whose length is the number of strings of 
        %         consecutive calculated singular values in svalp 
        %         (and svals). nsvalp(i) contains the number of calculated
        %         singular values in the ith string of consecutive 
        %         calculated singular values
        %   isvals: a vector whose length is the number of strings of
        %         consecutive calculated singular values in svals 
        %         (and svalp).  isvals(i) contains the starting index in
        %         svals of the ith string of consecutive calculated
        %         singular values.  Therefore the calculated singular
        %         values in svals (i.e. those not set equal to -1) have
        %         indices isvals(i): isvals(i) + nsvals(i) - 1 
        %         for i = 1: length(isvals)
        %   AND plot of partial singular value spectrum

       % have A, svals and svals_err from earlier code
        %[m,n]=size(A);
        isv = find( svals >= 0 );
        disv = diff(isv);
        il1 = find( disv > 1) ;
        k_m1=3;

        %create svalp and svalp_err
        clear isvals isvalp nsvalp
        svalp = [];
        svalp_err = [];
        for i = 1:length(il1)
           if ( i == 1 )
              isvalsv = isv(1):isv(il1(i));
              isvals(i) = isvalsv(1);
              if ( isv(1) == 1 )
                 svalp = svals(isvalsv);
                 svalp_err = svals_err(isvalsv);
                 isvalp(1)=1;
                 nsvalp(1) = isv(il1(i));
              else
                 svalp = [ - ones(k_m1,1) ;svals(isvalsv) ];
                 svalp_err = [ - ones(k_m1,1) ;svals_err(isvalsv) ];
                 isvalp(1) = k_m1+1;
                 nsvalp(1) = length(isvalsv);
              end
           else
              isvalp(i) = length(svalp)+1;
              isvalsv = isv(il1(i-1) + 1): isv(il1(i));
              isvals(i) = isvalsv(1);
              nsvalp(i) = length( svals( isv(il1(i-1) + 1): isv(il1(i))) );
              svalp = [ svalp; svals( isv(il1(i-1) + 1): isv(il1(i)))];
              svalp_err = [ svalp_err; svals_err( isv(il1(i-1) + 1): ...
                            isv(il1(i)))];
           end
           svalp = [svalp; - ones(k_m1,1) ];
           svalp_err = [svalp_err; - ones(k_m1,1) ];
        end
        % create isvalp, nsvalp, isvals and finish svalp
        if ( length( il1 ) == 0 )
           ib =1;
        else
           ib = isv( il1( length( il1 )) + 1 );
        end
        isvalp(length(il1)+1) = length(svalp) + 1;
        nsvalp(length(il1)+1) = length(ib:isv(end));
        isvalsv = ib :isv(end) ;
        isvals(length(il1)+1) = isvalsv(1);
        svalp = [ svalp ; svals( ib :isv(end)) ];
        svalp_err = [ svalp_err ; svals_err( ib :isv(end)) ];
        if(svals(end) < 0 )
           svalp = [svalp; - ones(k_m1,1)];
           svalp_err = [svalp_err; - ones(k_m1,1)];
        end

        if ( 1 == 0 )
           % display results (for debugging)
           disp('svals'''), disp(svals'),  disp('isv''')
           disp(isv'), disp('il1'''),  disp(il1')
           disp('svalp'''),  disp(svalp'), disp('svalp_err''')
           disp(svalp_err'), disp('isvalp'), disp(isvalp)
           disp('nsvalp'), disp(nsvalp), disp('isvals')
           disp(isvals)
        end

        %draw plots
        %warning('off','MATLAB:Axes:NegativeDataInLogAxis')
        if ( option == 2)
            % create svd image as thumbnail
            warning('off','MATLAB:Axes:NegativeDataInLogAxis')
            
            semilogy(svalp,'--rs','LineWidth',10,'MarkerEdgeColor','g', ...
                'MarkerFaceColor','g','MarkerSize',20);
            xlim([0 length(svalp)]);
            line([0 length(svalp)], [1 1]*tol, 'Color', [0 0 .5]); 
                                          % marker for tolerance
            set(gca,'XTick',[],'YTick',[]);
            
            axisv1 = axis;   % in Matlab 7.6 these 2 commands suppress
            axis(axisv1);    % the printing of an (undesired) warning
                             % reason is unclear  -- hmmm

            warning('on','MATLAB:Axes:NegativeDataInLogAxis');
                    
         else
            % prepare full-res svd image
            warning('off','MATLAB:Axes:NegativeDataInLogAxis')
            
            semilogy(svalp,'--rs','LineWidth',5,'MarkerEdgeColor','g', ...
                'MarkerFaceColor','g','MarkerSize',8);

            
            xlim([0 length(svalp)]);
            % finish full-res svd image
            colorm = [0 .5 0];
            line(xlim, [1 1]*tol, 'Color', colorm,'Linewidth',1); 
                               % marker for rank with MATLAB tolerance
               
            % fixes missing marker in Okunbor/aft01 plot
            axisv1 = axis;
            axisv1(4) = max( max(svalp), axisv1(4) );
            axis(axisv1);

            set(gca, 'FontSize', 17);
            %define locations to place tick marks
            xtick_v=[];
            for ixl = 1: length(isvalp)
               xtick_v = [ xtick_v, isvalp(ixl): ...
                           (isvalp(ixl)+nsvalp(ixl)-1) ];
            end
            if ( xtick_v(end) ~= length(svalp) )
               xtick_v =[ xtick_v length(svalp)];
            end   
            set(gca, 'XTick', xtick_v);

            %create labels for x axis
            xticklabelv = [];
            ixt = 0;
            for ixl = 1:length(isvals)
               for ixl2 = 1:nsvalp(ixl)
                 ixt = ixt+1;
                 if ( (ixl == 1 && nsvalp(ixl) < 10) | ixl2 == 1 | ...
                       ixl2 == nsvalp(ixl) )
                    xticklabelv{ixt}=int2str(isvals(ixl)+ixl2-1);
                 else 
                    xticklabelv{ixt} ='';
                 end
               end
            end
            ixt = ixt+1;
            xticklabelv{ixt} = int2str(min(m,n));
            set(gca, 'XTickLabel', xticklabelv);

            % finish full-res svd image
            grid on
            if default_tol
               text(1, tol, ' max(size(A)) * eps(max(svals))', 'Color',...
                    colorm,'FontSize', 14, 'VerticalAlignment', 'bottom');
            else
               text(1, tol,['tol = ',num2str(tol,'%0.5g')], 'Color',...
                   colorm,'FontSize', 14, 'VerticalAlignment', 'bottom');
            end
            title(['Partial Singular Value Spectrum for ' fullname],...
                 'FontSize',14, 'FontWeight', 'b', 'Interpreter', 'none') ;
            axisv = axis;

            % include errbounds in plot
            isvp=find(svalp >= 0 );
            isvp = isvp(:)';
            bound_lw = 1;   % width of line in error bounds
            for it = isvp
                if ( svalp_err(it) > 0 )
                  lowbound = max( axisv(3),svalp(it)-svalp_err(it));
                  upbound = min(axisv(4),svalp(it)+svalp_err(it));
                  line([it,it],[lowbound,upbound],'Color','b',...
                       'Linewidth',bound_lw);
                  boundlength = .15;
                  if ( lowbound ~= axisv(3) )
                     line([it - boundlength,it + boundlength],...
                          [lowbound,lowbound],...
                          'Color','b','Linewidth',bound_lw);
                  end 
                  if (upbound ~= axisv(4) )
                     line([it - boundlength, it+boundlength],...
                         [upbound, upbound],...
                         'Color','b','Linewidth',bound_lw);
                  end
                end
            end

            nsvals_zero = sum( svals == 0 ) ;
            if ( nsvals_zero == 1 )
               xlabel(['                       singular value ',...
                      'number \newline', ...
                      ' Note: one calculated singular value is 0 ',...
                      'and is not shown'],'FontSize', 14);         
            elseif (nsvals_zero > 1 )
               xlabel(['                          singular value ',...
                       'number \newline ',...
                       ' Note: ', int2str(nsvals_zero), ...
                       ' calculated singular values are 0 ', ...
                       'and are not shown'],'FontSize', 14);         
            else
                    xlabel('singular value number', 'FontSize', 14);
            end
            ylabel('calculated singular value', 'FontSize', 14);
            warning('on','MATLAB:Axes:NegativeDataInLogAxis')
        end
    end
end 
