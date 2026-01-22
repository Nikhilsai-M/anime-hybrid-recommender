import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Pointing to our FastAPI backend
const API_URL = "http://localhost:8000";

function App() {
  // State: managing user session and navigation
  const [userId, setUserId] = useState("100"); // Defaulting to User 100 for demo purposes
  const [view, setView] = useState("home");    // Simple router: 'home' | 'search' | 'watch'
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  // Data: storing the lists of anime
  const [homeFeed, setHomeFeed] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  
  // The specific anime selected for the "Watch" page
  const [selectedAnime, setSelectedAnime] = useState(""); 
  const [recs, setRecs] = useState({ content: [], ai: [] });

  // ---------------------------------------------------------
  // 1. INITIAL LOAD
  // Fetch the personalized "Top Picks" feed whenever the User ID changes.
  // ---------------------------------------------------------
  useEffect(() => {
    const fetchHome = async () => {
      setLoading(true);
      try {
        console.log(`Fetching home feed for User ${userId}...`);
        const res = await axios.get(`${API_URL}/home/${userId}`);
        setHomeFeed(res.data);
      } catch (e) { 
        console.error("Failed to fetch home feed", e); 
      }
      setLoading(false);
    };
    fetchHome();
  }, [userId]);

  // ---------------------------------------------------------
  // 2. SEARCH HANDLER
  // Sends the text query to BERT to find semantic matches.
  // ---------------------------------------------------------
  const handleSearch = async () => {
    if (!query) return;
    
    setLoading(true);
    setView("search"); // Switch view immediately
    
    try {
      const res = await axios.post(`${API_URL}/search`, { 
        user_id: userId, 
        query: query 
      });
      setSearchResults(res.data.error ? [] : res.data);
    } catch (e) { 
      console.error("Search failed", e); 
    }
    setLoading(false);
  };

  // ---------------------------------------------------------
  // 3. WATCH & RECOMMEND HANDLER
  // When a user clicks a card, we simulate "watching" it and 
  // generate recommendations based on that specific anime.
  // ---------------------------------------------------------
  const handleAnimeClick = async (animeTitle) => {
    setLoading(true);
    setSelectedAnime(animeTitle);
    setView("watch"); // Switch to detail view
    
    // Scroll to top for a cinematic feel
    window.scrollTo({ top: 0, behavior: 'smooth' });

    try {
      const res = await axios.post(`${API_URL}/recommend`, { 
        user_id: userId, 
        query: animeTitle 
      });
      setRecs(res.data);
    } catch (e) { 
      console.error("Recommendation failed", e); 
    }
    setLoading(false);
  };

  return (
    <div className="container">
      
      {/* --- GLOBAL HEADER --- */}
      <div className="header">
        <h1 onClick={() => setView('home')}>🍿 AnimeStream</h1>
        
        {/* Search Bar */}
        <div className="input-group">
          <input 
            value={query} 
            onChange={(e) => setQuery(e.target.value)} 
            placeholder="Search for an anime (e.g., 'Time travel thriller')..." 
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button onClick={handleSearch}>Search</button>
        </div>

        {/* User ID Switcher (For Demoing Personalization) */}
        <input 
          value={userId} 
          onChange={(e) => setUserId(e.target.value)} 
          style={{width: '60px', textAlign: 'center'}}
          title="Simulate different users"
        />
      </div>

      {/* --- LOADING STATE --- */}
      {loading && (
        <div style={{textAlign:'center', padding:'100px', fontSize:'1.5rem', color:'#94a3b8'}}>
          ✨ Curating Intelligence...
        </div>
      )}

      {/* --- VIEW 1: HOME FEED --- */}
      {/* Shows the Neural Network's predictions for the current user */}
      {!loading && view === 'home' && (
        <div>
          <h2 className="section-title">🔥 Top Picks For You</h2>
          <div className="grid">
            {homeFeed.map((item) => (
              <div key={item.id} className="card" onClick={() => handleAnimeClick(item.title)}>
                <img src={item.image} alt={item.title} loading="lazy" />
                <div className="card-content">
                  <h3>{item.title}</h3>
                  <div className="score match-green">
                    {Math.round(item.score * 100)}% Match
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* --- VIEW 2: SEARCH RESULTS --- */}
      {/* Shows semantic matches from BERT */}
      {!loading && view === 'search' && (
        <div>
           <button className="back-btn" onClick={() => setView('home')}>← Back Home</button>
           <h2 className="section-title">🔍 Results for "{query}"</h2>
           
           <div className="grid">
            {searchResults.length === 0 ? <p style={{color:'#94a3b8'}}>No results found.</p> : 
              searchResults.map((item) => (
                <div key={item.id} className="card" onClick={() => handleAnimeClick(item.title)}>
                  <img src={item.image} alt={item.title} />
                  <div className="card-content">
                    <h3>{item.title}</h3>
                    <div className="score match-green">
                      {Math.round(item.match_score * 100)}% Similarity
                    </div>
                  </div>
                </div>
              ))
            }
           </div>
        </div>
      )}

      {/* --- VIEW 3: WATCH PAGE --- */}
      {/* The core "Hybrid" demo: Comparing Content-based vs AI-based recs */}
      {!loading && view === 'watch' && (
        <div>
           <button className="back-btn" onClick={() => setView('search')}>← Back to Search</button>
           
           {/* Cinematic Hero Section */}
           <div className="hero">
             <h1>{selectedAnime}</h1>
             <p>Now Playing • Hybrid AI Analysis Complete</p>
           </div>

           {/* Row 1: The "Objective" Recommendations (Content-Based) */}
           <h3 className="section-title">Because you watched {selectedAnime}</h3>
           <div className="netflix-row">
             {recs.content.map((item) => (
               <div key={item.id} className="card card-scroll" onClick={() => handleAnimeClick(item.title)}>
                 <img src={item.image} alt={item.title} />
                 <div className="card-content">
                   <h3>{item.title}</h3>
                   {/* Displaying raw similarity score from BERT */}
                   <div className="score match-green">
                     {Math.round(item.match * 100)}% Plot Match
                   </div>
                 </div>
               </div>
             ))}
           </div>

           {/* Row 2: The "Personalized" Recommendations (Neural Network) */}
           <h3 className="section-title" style={{marginTop:'60px'}}>
             🧠 AI Picks for User {userId} <span style={{fontSize:'0.8rem', color:'#60a5fa'}}>(Personalized)</span>
           </h3>
           
           <div className="netflix-row">
             {recs.ai.map((item) => (
               <div key={item.id} className="card card-scroll" onClick={() => handleAnimeClick(item.title)} style={{border:'1px solid #3b82f6'}}>
                 <img src={item.image} alt={item.title} />
                 <div className="card-content">
                   <h3>{item.title}</h3>
                   {/* Displaying the weighted Hybrid Score */}
                   <div className="score match-blue">
                     {Math.round(item.hybrid * 100)}% For You
                   </div>
                 </div>
               </div>
             ))}
           </div>
        </div>
      )}

    </div>
  );
}

export default App;