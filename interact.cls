%%
%% This is file `interact.cls',
%% Taylor & Francis Interact layout LaTeX2e document class.
%% Compatible with pdfLaTeX, XeLaTeX, and LuaLaTeX.
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{interact}[2021/02/02 v1.06 Taylor & Francis Interact layout document class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper,oneside]{article}

\RequirePackage{geometry}
\geometry{
  a4paper,
  textwidth=150mm,
  textheight=230mm,
  centering,
  headheight=14pt,
  headsep=18pt,
  footskip=28pt
}

\RequirePackage{etoolbox}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{caption}

% Caption formatting per Taylor & Francis Interact style
\captionsetup[table]{
  labelfont=bf,
  textfont=normalfont,
  labelsep=period,
  justification=raggedright,
  singlelinecheck=false,
  position=top,
  skip=6pt
}
\captionsetup[figure]{
  labelfont=bf,
  textfont=normalfont,
  labelsep=period,
  justification=raggedright,
  singlelinecheck=false,
  position=bottom,
  skip=6pt
}

% Support standard T&F \tbl{caption}{table body} macro as well as standard \caption
\newcommand{\tbl}[2]{%
  \caption{#1}%
  #2%
}
\newcommand{\tabnote}[1]{%
  \par\vspace{3pt}{\footnotesize #1\par}%
}

% Article metadata macros
\def\@articletype{RESEARCH ARTICLE}
\newcommand{\articletype}[1]{\def\@articletype{#1}}

\def\@received{}
\newcommand{\received}[1]{\def\@received{#1}}

\newcommand{\name}[1]{{\large\bfseries #1}\par\vspace{4pt}}
\newcommand{\affil}[1]{{\small\itshape #1}\par}

% Redefine \maketitle for Taylor & Francis Interact layout
\renewcommand{\maketitle}{%
  \par
  \begingroup
    \renewcommand\thefootnote{\fnsymbol{footnote}}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \newpage
    \global\@topnum\z@
    \@maketitle
    \thispagestyle{plain}%
    \@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@maketitle{%
  \newpage
  \null
  \vspace*{-12pt}%
  \begin{flushleft}%
    \ifx\@articletype\@empty\else
      {\footnotesize\bfseries\MakeUppercase{\@articletype}\par}%
      \vspace{8pt}%
    \fi
    {\LARGE\bfseries \@title \par}%
    \vspace{14pt}%
    {\normalsize
      \lineskip .5em%
      \@author
    }%
    \ifx\@received\@empty\else
      \vspace{8pt}%
      {\footnotesize \@received \par}%
    \fi
  \end{flushleft}%
  \par
  \vspace{10pt}%
  \hrule height 0.5pt
  \vspace{10pt}%
}

% Abstract and Keywords environments in Taylor & Francis Interact style
\renewenvironment{abstract}{%
  \small
  \noindent\textbf{ABSTRACT}\par\vspace{4pt}%
  \noindent\ignorespaces
}{%
  \par\vspace{8pt}%
}

\newenvironment{keywords}{%
  \small
  \noindent\textbf{KEYWORDS}\par\vspace{2pt}%
  \noindent\ignorespaces
}{%
  \par\vspace{12pt}%
  \hrule height 0.5pt
  \vspace{14pt}%
}

% Section formatting per Taylor & Francis Interact guidelines
\renewcommand\thesection{\arabic{section}.}
\renewcommand\thesubsection{\thesection\arabic{subsection}.}
\renewcommand\thesubsubsection{\thesubsection\arabic{subsubsection}.}

\renewcommand\section{\@startsection {section}{1}{\z@}%
  {-3.25ex \@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-2.75ex\@plus -1ex \@minus -.2ex}%
  {1.2ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries\itshape}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-2.25ex\@plus -1ex \@minus -.2ex}%
  {1.0ex \@plus .2ex}%
  {\normalfont\normalsize\itshape}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
  {1.5ex \@plus0.5ex \@minus.2ex}%
  {-1em}%
  {\normalfont\normalsize\itshape}}

% Theorem styles per Taylor & Francis Interact guidelines
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}

\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{notation}{Notation}

\endinput

