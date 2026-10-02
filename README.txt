Baggrund forud for kommende møde vedr. AI-sparring i Patologi

Vi er i gang med at digitalisere cytologi på AUH (undersøgelse af enkeltceller fra fx nålebiopsier og væsker). I den forbindelse vil vi gerne undersøge mulighederne for at udvikle simple AIstøtteværktøjer, der kan forbedre workflow og effektivitet.
Det er vigtigt at understrege, at vi ikke er interesserede i avancerede “black box” diagnosemodeller.Vi ønsker i stedet en pragmatisk “light AI” tilgang, hvor AI hjælper med strukturering og prioritering – ikke selvstændig diagnostik.

Overordnet mål
At udvikle AIfunktioner der:
	•	reducerer manuelt arbejde
	•	gør digitale præparater lettere at gennemgå
	•	forbedrer triagering og kvalitet


Case 1: Lokalisering af celleområder (“Scopio-light”)

Formål:Gøre cytologipræparater lettere og hurtigere at gennemgå digitalt. Undgå at patologen skal bruge lang tid på at screene glasset.

Idé:En algoritme der når præparatet er scannet:
finder områder med celler, mucin, kolloid osv. (vs tom baggrund/blod)
opdeler billedet i mindre felter (fx grid)
viser disse som “regions of interest”

Output (for bruger):
markerede områder med relevant indhold
evt. prioriteret gennemgang

Vigtigt:
ingen klassifikation af celletype
ingen diagnoser
kun “hvor er der noget at kigge på”

Gevinst:
mindre spildtid på tomme områder
mere ensartet gennemgang
bedre digital screening


Case 2: Triagering (egnede vs uegnede prøver)

Formål:Automatisk sortering og kvalitetssikring af prøver

Idé:En algoritme der vurderer:
	•	cellemængde
	•	baggrundsstøj (blod, debris)
	•	fokus/kvalitet
og klassificerer prøver som:
	•	egnede
	•	suboptimale
	•	uegnede

Gevinst:
færre ressourcer brugt på uegnede præparater
bedre prioritering
uegnede afsluttes med standardtekst (meget lidt hands-on tid)
mulighed for feedback til klinik

Hvorfor denne tilgang?
Cytologi er meget variabelt og teknisk komplekst
Datagrundlag er begrænset
De største gevinster ligger i workflowoptimering – ikke automatiseret diagnose

Teknisk tilgang (overordnet)
Vi forestiller os:
	•	Fokus på billedanalyse (detektion/segmentering)
	•	Ikke avanceret klassifikation i første omgang
	•	Integration i kommende digitale billedsystemer (ikke separat løsning)
 
Ambitionsniveau
Kort sigt:
	•	simple modeller til cellelokalisering og triagering
Mellemlang sigt:
	•	forbedret scoring og prioritering
Lang sigt (evt.):
	•	mere avanceret analyse

Vi vil gerne udvikle:
simple, robuste AIværktøjer der hjælper os med at arbejde smartere.


