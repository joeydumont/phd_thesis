% --------------------------------------------------------------------------- %
% Author:          Joey Dumont                <joey.dumont@gmail.com>         %
% Date created:    Mar. 8th, 2017                                             %
% Description:     INRS thesis class file. Based on the memoir class, with    %
%                  elements of the UL thesis class, ulthese.                  %
% License:         CC0                                                        %
%                  <https://creativecommons.org/publicdomain/zero/1.0>        %
% --------------------------------------------------------------------------- %

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{inrsthesis}[2017/03/08 v1.0 INRS thesis class]

% -- Pass other options directly to memoir class.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax

\LoadClass{memoir}

