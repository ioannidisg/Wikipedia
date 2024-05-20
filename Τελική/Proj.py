import networkx as nx
import wikipedia
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# για να γίνουν ignore warnings που δεν επιρεάζουν το αποτέλεσμα ούτε τη λειτουργία του προγράμματος
warnings.catch_warnings()
warnings.simplefilter("ignore")

lim = 10  # Για το πόσο μεγάλος θα είναι ο γράφος (πόσα Links)

wikipedia.set_lang('en')  # θέτω τη γλώσσα των άρθρων που θα ψάχνω στη wikipedia ως Αγγλική

result = wikipedia.search("Graph theory")  # Αναζητώ άρθρα απο τη wikipedia που σχετίζονται με το "Graph theory"
print(result)  # τυπώνω απλά για να πάρω μια ιδέα

# Παίρνω το πρώτο άρθρο
arthro1_title = result[0]
arthro1 = wikipedia.page(arthro1_title)

links = arthro1.links[:lim]  # Παίρνω τους συνδέσμους από το πρώτο άρθρο (βάζω το 10 ως όριο)
print("Links for", arthro1_title)
print(links)

# Ορίζω ένα μοντέλο για να υπολογίσω τη σημασιολογικής συσχέτισης
model = SentenceTransformer('all-MiniLM-L6-v2')

G = nx.DiGraph()  # Δημιουργώ έναν κενό

G.add_node(arthro1_title)   # Βάζω το πρώτο άρθρο ως κόμβο στον γράφο
arthro1_text = arthro1.content  # Παίρνω το περιεχόμενο του άρθρου

arthro1_embedding = model.encode([arthro1_text])[0]  # Αναπαράσταση του πρώτου άρθρου (το κάνει με πολλές παραμέτρους)

# Για κάθε σύνδεσμο στο πρώτο άρθρο
for link in links:
    try:
        linked_page = wikipedia.page(link, auto_suggest=False)
        linked_text = linked_page.content

        # Υπολογίζω την αναπαράσταση του συνδέσμου
        linked_embedding = model.encode([linked_text])[0]

        # Σημασιολογική ομοιότητα μεταξύ του πρώτου άρθρου και του συνδέσμου
        s = float(cosine_similarity([arthro1_embedding], [linked_embedding])[0][0])
        G.add_edge(arthro1_title, link, weight=s)
        # Βάζω τη σύνδεση με τον σύνδεσμο στον γράφο και το βάρος που είναι η σημασιολογική συσχέτιση s.

        linked_links = linked_page.links[:lim]  # Παίρνω τους συνδέσμους του συνδέσμου (ξανά το 10 ως όριο)

        # Για κάθε σύνδεσμο στον σύνδεσμο
        for linked_link in linked_links:
            try:
                linked_link_text = wikipedia.page(linked_link, auto_suggest=False).content
                linked_link_embedding = model.encode([linked_link_text])[0]

                # Υπολογίζουμε τη σημασιολογική ομοιότητα μεταξύ του συνδέσμου και του κάθε συνδέσμου του
                s = float(cosine_similarity([linked_embedding], [linked_link_embedding])[0][0])

                # Προσθέτουμε τη σύνδεση μεταξύ του συνδέσμου και του κάθε συνδέσμου του
                G.add_edge(link, linked_link, weight=s)
            except wikipedia.exceptions.DisambiguationError:
                # Αγνοώ τους συνδέσμους που δεν είναι σαφείς
                continue
    except wikipedia.exceptions.DisambiguationError:
        # Αγνοώ τους συνδέσμους που δεν είναι σαφείς
        continue

print("Nodes:", G.nodes())
print("Edges:", G.edges())
nx.write_graphml(G, r"C:\Users\ioann\OneDrive\Desktop\Project_Dir.graphml")
 #  Χρειάζεται τον χρόνο του να τρέξει     .
 #  Δημιουργεί το αρχείο του γράφου στο dekstop μου άρα αν θελετε να το τρέξετε πρέπει να αλλάξετε το path της τελευταίας ετνολής