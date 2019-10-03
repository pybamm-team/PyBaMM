function varargout = shellgui(varargin)
%SHELLGUI GUI interface for seashell function
%   Timothy A. Davis, Chapman Hall / CRC Press, 7th edition.
%   Controls the parameters a, b, c, n, azimuth, and elevation, using
%   sliders.  To the whole range of each parameter, click on the button to
%   the right of each slider.
%
% Example:
%   shellgui
%
% See also GUIDE, SEASHELL

% Copyright 2006 Timothy A. Davis

% Last Modified by GUIDE v2.5 29-Jul-2006 11:33:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @shellgui_OpeningFcn, ...
                   'gui_OutputFcn',  @shellgui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT



% --- Executes just before shellgui is made visible.
function shellgui_OpeningFcn(hObject, eventdata, handles, varargin) %#ok
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to shellgui (see VARARGIN)

% Choose default command line output for shellgui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes shellgui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = shellgui_OutputFcn(hObject, eventdata, handles) %#ok
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles) %#ok
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global a b c n azimuth elevation
a = get (hObject, 'Value') ;
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)         %#ok
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)          %#ok
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global a b c n azimuth elevation
b = get (hObject, 'Value') ;
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)         %#ok
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

global a b c n azimuth elevation
a = -0.2 ;
b = 0.5 ;
c = 0.1 ;
n = 2 ;
azimuth = -150 ;
elevation = 10 ;

seashell ;



% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)          %#ok
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

global a b c n azimuth elevation
c = get (hObject, 'Value') ;
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)         %#ok
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)          %#ok
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


global a b c n azimuth elevation
n = get (hObject, 'Value') ;
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)         %#ok
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes on slider movement.
function slider8_Callback(hObject, eventdata, handles)          %#ok
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


global a b c n azimuth elevation
azimuth = get (hObject, 'Value') ;
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes during object creation, after setting all properties.
function slider8_CreateFcn(hObject, eventdata, handles)         %#ok
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider9_Callback(hObject, eventdata, handles)          %#ok
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


global a b c n azimuth elevation
elevation = get (hObject, 'Value') ;
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes during object creation, after setting all properties.
function slider9_CreateFcn(hObject, eventdata, handles)         %#ok
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global a b c n azimuth elevation
seashell (a, b, c, n, Inf, elevation) ;
seashell (a, b, c, n, azimuth, elevation) ;


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global a b c n azimuth elevation
for a2 = -1:.1:1
    seashell (a2, b, c, n, azimuth, elevation) ;
    drawnow
end
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global a b c n azimuth elevation
for b2 = -1:.1:1
    seashell (a, b2, c, n, azimuth, elevation) ;
    drawnow
end
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


global a b c n azimuth elevation
for c2 = -1:.1:1
    seashell (a, b, c2, n, azimuth, elevation) ;
    drawnow
end
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global a b c n azimuth elevation
for n2 = 0:.5:8
    seashell (a, b, c, n2, azimuth, elevation) ;
    drawnow
end
seashell (a, b, c, n, azimuth, elevation) ;

% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


global a b c n azimuth elevation
for el = -80:10:80
    seashell (a, b, c, n, azimuth, el) ;
    drawnow
end
seashell (a, b, c, n, azimuth, elevation) ;


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)      %#ok
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

web ('http://www.suitesparse.com') ;

